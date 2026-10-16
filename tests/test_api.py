"""Macro&Micro regression тест (stdlib unittest).

Ажиллуулах:
    .venv-pathmnist/bin/python -m unittest discover -s tests -v

Тест бүр түр зуурын SQLite файл ашиглана — db/macro2micro.db-д хүрэхгүй.
"""

import ast
import io
import os
import shutil
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
TMP_DIR = Path(tempfile.mkdtemp(prefix="macro2micro-test-"))
os.environ["DATABASE_PATH"] = str(TMP_DIR / "test.db")

import app as app_module  # noqa: E402  (DATABASE_PATH тохируулсны дараа import)
import config  # noqa: E402
from services import business_logic, database, model_loader  # noqa: E402

PATHMNIST_DEBRIS = ROOT / "pathmnist_2_debris.png"
PATHMNIST_ADENO = ROOT / "pathmnist_8_adenocarcinoma.png"
TOXO_CARDIAC = ROOT / "images" / "histology_heart" / "cardiac_bradyzoite.png"
TOXO_MYOCARDITIS = ROOT / "images" / "histology_heart" / "myocarditis_high.png"
TOXO_BRAIN_GROSS = ROOT / "images" / "Gross_images" / "Brain_001.png"

LEGACY_FIELDS = {"class", "label", "confidence", "notice"}
NEW_FIELDS = {"decision", "decision_message", "explanation", "dataset", "analysis_id"}


def tearDownModule():
    shutil.rmtree(TMP_DIR, ignore_errors=True)


class ApiTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app_module.app.test_client()

    def post_file(self, endpoint, path=None, data=None, filename=None):
        if path is not None:
            data = path.read_bytes()
            filename = filename or path.name
        return self.client.post(
            endpoint,
            data={"image": (io.BytesIO(data), filename)},
            content_type="multipart/form-data",
        )

    def db_rows(self, query, params=()):
        connection = sqlite3.connect(config.DATABASE_PATH)
        connection.row_factory = sqlite3.Row
        try:
            return [dict(row) for row in connection.execute(query, params)]
        finally:
            connection.close()


class TestStartupAndUi(ApiTestCase):
    def test_home_serves_index_with_existing_ids(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        for element_id in [
            "ai-image", "ai-preview", "ai-preview-container", "ai-predict-button",
            "ai-result", "pathmnist-result", "pathology-learning", "related-macro",
            "ai-reset-button", "ai-disclaimer", "ai-results-area",
        ]:
            self.assertIn(f'id="{element_id}"', html, element_id)
        for removed in ("welcome-name", "welcome-screen", "welcome-next"):
            self.assertNotIn(f'id="{removed}"', html, "нэрийн welcome flow устгагдсан байх ёстой")
        self.assertIn('fetch("/predict"', html)
        self.assertIn('fetch("/predict-pathology"', html)
        response.close()

    def test_gross_image_route(self):
        response = self.client.get("/images/Gross_images/Heart_001.png")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "image/png")
        response.close()


class TestValidation(ApiTestCase):
    def test_missing_file(self):
        for endpoint in ("/predict", "/predict-pathology"):
            response = self.client.post(endpoint, data={})
            self.assertEqual(response.status_code, 400)
            self.assertIn("error", response.get_json())

    def test_empty_filename(self):
        response = self.post_file("/predict", data=b"", filename="")
        self.assertEqual(response.status_code, 400)

    def test_invalid_extension(self):
        for filename in ("notes.txt", "scan.gif", "image.png.exe", "noext"):
            response = self.post_file("/predict", data=b"hello", filename=filename)
            self.assertEqual(response.status_code, 415, filename)
            self.assertIn("error", response.get_json())

    def test_fake_png_rejected(self):
        response = self.post_file("/predict-pathology", data=b"not an image", filename="pathmnist_fake.png")
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_too_large_upload(self):
        big = b"0" * (config.MAX_UPLOAD_BYTES + 1024)
        response = self.post_file("/predict", data=big, filename="big.png")
        self.assertEqual(response.status_code, 413)
        self.assertIn("error", response.get_json())

    def test_invalid_upload_not_logged(self):
        before = self.db_rows("SELECT COUNT(*) AS n FROM analyses")[0]["n"]
        self.post_file("/predict", data=b"hello", filename="notes.txt")
        after = self.db_rows("SELECT COUNT(*) AS n FROM analyses")[0]["n"]
        self.assertEqual(before, after)


class TestPredictions(ApiTestCase):
    def assert_compatible(self, body):
        self.assertTrue(LEGACY_FIELDS <= body.keys(), body.keys())
        self.assertTrue(NEW_FIELDS <= body.keys(), body.keys())
        self.assertIn(body["decision"], ("classified", "low_confidence"))
        self.assertGreaterEqual(body["confidence"], 0)
        self.assertLessEqual(body["confidence"], 100)

    def test_pathmnist_flow(self):
        r1 = self.post_file("/predict", PATHMNIST_ADENO)
        r2 = self.post_file("/predict-pathology", PATHMNIST_ADENO)
        self.assertEqual(r1.status_code, 200)
        self.assertEqual(r2.status_code, 200)

        image_type, tissue = r1.get_json(), r2.get_json()
        self.assert_compatible(image_type)
        self.assert_compatible(tissue)
        self.assertEqual(image_type["class"], "micro")
        self.assertEqual(image_type["label"], "Микроскопийн зураг")
        self.assertEqual(image_type["dataset"], "PathMNIST")
        self.assertEqual(tissue["class"], 8)
        self.assertEqual(tissue["label"], "Colorectal adenocarcinoma epithelium")
        self.assertEqual(tissue["decision"], "classified")
        self.assertIsNone(tissue["decision_message"])
        self.assertIn("morphology", tissue)

    def test_low_confidence_decision(self):
        # Энэ зураг дээр CNN ~59% магадлалтай → threshold 0.60-аас бага
        tissue = self.post_file("/predict-pathology", PATHMNIST_DEBRIS).get_json()
        self.assertEqual(tissue["decision"], "low_confidence")
        self.assertEqual(tissue["decision_message"], business_logic.LOW_CONFIDENCE_MESSAGE)

    def test_toxopath_flow(self):
        for path in (TOXO_CARDIAC, TOXO_MYOCARDITIS, TOXO_BRAIN_GROSS):
            response = self.post_file("/predict", path)
            self.assertEqual(response.status_code, 200, path.name)
            body = response.get_json()
            self.assert_compatible(body)
            self.assertEqual(body["dataset"], "ToxoPath-1")
        self.assertEqual(self.post_file("/predict", TOXO_BRAIN_GROSS).get_json()["class"], "macro")

    def test_models_loaded_once(self):
        self.post_file("/predict", PATHMNIST_ADENO)
        with mock.patch.object(model_loader, "_load_pickle") as load_pickle, \
             mock.patch.object(model_loader, "_load_keras") as load_keras:
            self.post_file("/predict", PATHMNIST_ADENO)
            self.post_file("/predict-pathology", PATHMNIST_ADENO)
            load_pickle.assert_not_called()
            load_keras.assert_not_called()

    def test_db_failure_does_not_break_prediction(self):
        with mock.patch.object(database, "save_analysis", side_effect=sqlite3.OperationalError("locked")):
            response = self.post_file("/predict", TOXO_CARDIAC)
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.get_json()["analysis_id"])


class TestDatabase(ApiTestCase):
    def test_database_file_and_tables(self):
        self.assertTrue(Path(config.DATABASE_PATH).exists())
        tables = {row["name"] for row in self.db_rows("SELECT name FROM sqlite_master WHERE type = 'table'")}
        self.assertTrue({"analyses", "predictions"} <= tables)

    def test_foreign_key_declared(self):
        fks = self.db_rows("PRAGMA foreign_key_list(predictions)")
        self.assertEqual(len(fks), 1)
        self.assertEqual(fks[0]["table"], "analyses")
        self.assertEqual(fks[0]["from"], "analysis_id")
        self.assertEqual(fks[0]["to"], "id")
        self.assertEqual(fks[0]["on_delete"], "CASCADE")

    def test_prediction_creates_linked_rows(self):
        body = self.post_file("/predict-pathology", PATHMNIST_ADENO).get_json()
        analysis_id = body["analysis_id"]
        self.assertIsInstance(analysis_id, int)

        saved = database.get_analysis(analysis_id)
        self.assertEqual(saved["dataset"], "PathMNIST")
        self.assertEqual(saved["file_ext"], ".png")
        self.assertEqual(len(saved["predictions"]), 1)
        prediction = saved["predictions"][0]
        self.assertEqual(prediction["analysis_id"], analysis_id)
        self.assertEqual(prediction["model_name"], "pathmnist_cnn")
        self.assertEqual(prediction["predicted_class"], "8")
        self.assertEqual(prediction["label"], body["label"])
        self.assertAlmostEqual(prediction["confidence"], body["confidence"])
        self.assertEqual(prediction["decision"], body["decision"])

    def test_orphan_prediction_rejected(self):
        connection = database.get_connection()
        try:
            with self.assertRaises(sqlite3.IntegrityError):
                with connection:
                    connection.execute(
                        database.INSERT_PREDICTION,
                        (999999, "m", "0", "x", 50.0, "classified"),
                    )
        finally:
            connection.close()

    def test_cascade_delete(self):
        analysis_id = database.save_analysis("PathMNIST", ".png", [
            {"model_name": "m", "predicted_class": 1, "label": "Background",
             "confidence": 99.0, "decision": "classified"},
        ])
        connection = database.get_connection()
        try:
            with connection:
                connection.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
        finally:
            connection.close()
        self.assertEqual(self.db_rows("SELECT * FROM predictions WHERE analysis_id = ?", (analysis_id,)), [])

    def test_sql_injection_stored_as_literal(self):
        payload = "x'); DROP TABLE analyses; --"
        analysis_id = database.save_analysis(payload, ".png", [])
        self.assertEqual(database.get_analysis(analysis_id)["dataset"], payload)
        self.assertTrue(self.db_rows("SELECT name FROM sqlite_master WHERE name = 'analyses'"))

    def test_no_pii_columns(self):
        columns = {
            row["name"]
            for table in ("analyses", "predictions")
            for row in self.db_rows("SELECT name FROM pragma_table_info(?)", (table,))
        }
        for forbidden in ("name", "user", "username", "filename", "image", "image_data"):
            self.assertNotIn(forbidden, columns)


class TestSqlSafety(unittest.TestCase):
    """Бүх execute(...)-ийн SQL аргумент тогтмол string/нэр байх ёстой."""

    SQL_METHODS = {"execute", "executemany", "executescript"}

    def python_files(self):
        for path in [ROOT / "app.py", ROOT / "config.py", *(ROOT / "services").glob("*.py")]:
            yield path

    def test_no_dynamic_sql(self):
        for path in self.python_files():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            constants = {
                target.id: node.value
                for node in tree.body if isinstance(node, ast.Assign)
                for target in node.targets if isinstance(target, ast.Name)
            }
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr in self.SQL_METHODS and node.args):
                    continue
                sql = node.args[0]
                if isinstance(sql, ast.Name) and sql.id in constants:
                    sql = constants[sql.id]
                if isinstance(sql, ast.Name):
                    continue  # жишээ нь schema файлаас уншсан script
                self.assertIsInstance(
                    sql, ast.Constant,
                    f"{path.name}:{node.lineno} SQL нь f-string/concatenation байна",
                )
                self.assertIsInstance(sql.value, str)


class TestBusinessLogic(unittest.TestCase):
    def test_detect_dataset(self):
        self.assertEqual(business_logic.detect_dataset("pathmnist_2_debris.png"), "PathMNIST")
        self.assertEqual(business_logic.detect_dataset("PathMNIST_8_x.PNG"), "PathMNIST")
        self.assertEqual(business_logic.detect_dataset("cardiac_bradyzoite.png"), "ToxoPath-1")
        self.assertEqual(business_logic.detect_dataset("my_pathmnist_1.png"), "ToxoPath-1")

    def test_threshold_boundary(self):
        self.assertEqual(business_logic.decide(0.5999)[0], "low_confidence")
        self.assertEqual(business_logic.decide(0.60)[0], "classified")

    def test_all_nine_classes(self):
        for class_id in range(9):
            result = business_logic.interpret_tissue(class_id, 0.9)
            self.assertTrue(result["label"] and result["morphology"] and result["explanation"])
        with self.assertRaises(ValueError):
            business_logic.interpret_tissue(9, 0.9)


class TestModelComparisonReport(unittest.TestCase):
    """train_compare_pathmnist.py-ийн гаралтыг шалгана (эхлээд script-ийг ажиллуулсан байх)."""

    REPORT = ROOT / "reports" / "model_comparison.json"

    @classmethod
    def setUpClass(cls):
        if not cls.REPORT.exists():
            raise unittest.SkipTest("reports/model_comparison.json байхгүй")
        import json
        cls.report = json.loads(cls.REPORT.read_text(encoding="utf-8"))

    def test_report_files_exist(self):
        from PIL import Image
        self.assertTrue((ROOT / "reports" / "model_comparison.md").exists())
        for name in ("confusion_matrix_cnn.png", "confusion_matrix_logreg.png"):
            with Image.open(ROOT / "reports" / name) as image:
                self.assertEqual(image.format, "PNG")

    def test_two_models_same_task_same_full_test_set(self):
        models = self.report["models"]
        self.assertEqual(len(models), 2)
        test_size = self.report["data"]["test_size"]
        self.assertEqual(test_size, 7180)
        self.assertFalse(self.report["data"]["test_set_subsampled"])
        for metrics in models.values():
            self.assertEqual(sum(row["support"] for row in metrics["per_class"]), test_size)
            self.assertEqual(sum(map(sum, metrics["confusion_matrix"])), test_size)
            self.assertEqual(len(metrics["per_class"]), 9)

    def test_no_leakage(self):
        leak = self.report["data"]["leakage_check"]
        self.assertEqual(leak["subset_test_exact_duplicates"], 0)
        self.assertEqual(leak["train_test_exact_duplicates"], 0)

    def test_metrics_valid_range(self):
        import math
        for metrics in self.report["models"].values():
            values = [metrics[k] for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1")]
            values += [row[k] for row in metrics["per_class"] for k in ("precision", "recall", "f1")]
            for value in values:
                self.assertTrue(math.isfinite(value) and 0.0 <= value <= 1.0, value)

    def test_winner_follows_metrics(self):
        models = self.report["models"]
        best = max(models, key=lambda name: (models[name]["macro_f1"], models[name]["accuracy"]))
        self.assertEqual(self.report["comparison"]["winner"], best)

    def test_model_b_file_loads_and_predicts(self):
        import joblib
        import numpy as np
        from PIL import Image
        model_b = joblib.load(ROOT / "pathmnist_logreg.pkl")
        pixels = np.array(Image.open(PATHMNIST_ADENO).convert("RGB"), dtype=np.float32) / 255.0
        prediction = int(model_b.predict(pixels.reshape(1, -1))[0])
        self.assertIn(prediction, range(9))


class TestFrontendRegression(unittest.TestCase):
    def test_frontend_mock_flow(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("node суугаагүй")
        result = subprocess.run(
            [node, str(ROOT / "tests" / "frontend_regression.js")],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
