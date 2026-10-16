"""Macro&Micro Flask API (UI serve + REST endpoints).

app.py нь зөвхөн route, validation болон service дуудалт хийнэ:
- services/validation.py     — upload шалгалт
- services/model_loader.py   — model-уудыг нэг удаа ачаалж inference хийх
- services/business_logic.py — model output → шийдвэр, тайлбар
- services/database.py       — SQLite logging (parameterized query)
"""

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge

import config
from services import business_logic, database, model_loader
from services.validation import ValidationError, too_large_message, validate_upload

BASE_DIR = config.BASE_DIR

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_BYTES

try:
    database.init_db()
except Exception:
    # DB асуудал AI prediction-ийг зогсоохгүй; logging л ажиллахгүй болно
    app.logger.exception("Database-ийг эхлүүлж чадсангүй.")


def error_response(message, status_code):
    return jsonify({"error": message}), status_code


def log_prediction(dataset, file_ext, model_name, result):
    """Prediction-ийг DB-д бичнэ. Алдаа гарвал None буцааж, response-ийг эвдэхгүй."""
    try:
        return database.save_analysis(
            dataset,
            file_ext,
            [{
                "model_name": model_name,
                "predicted_class": result["class"],
                "label": result["label"],
                "confidence": result["confidence"],
                "decision": result["decision"],
            }],
        )
    except Exception:
        app.logger.exception("Prediction-ийг database-д бичиж чадсангүй.")
        return None


@app.errorhandler(RequestEntityTooLarge)
def handle_too_large(_error):
    return error_response(too_large_message(), 413)


@app.get("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/images/<path:filename>")
def image_file(filename):
    return send_from_directory(BASE_DIR / "images", filename)


@app.post("/predict")
def predict():
    try:
        upload = validate_upload(request.files.get("image"))
    except ValidationError as error:
        return error_response(error.message, error.status_code)

    try:
        prediction, probability = model_loader.predict_image_type(upload.image)
    except model_loader.ModelNotFoundError:
        return error_response("AI загвар олдсонгүй.", 500)
    except Exception:
        app.logger.exception("/predict inference алдаа")
        return error_response("AI загвар зураг боловсруулахад алдаа гарлаа.", 500)

    result = business_logic.interpret_image_type(prediction, probability)
    result["dataset"] = business_logic.detect_dataset(upload.filename)
    result["analysis_id"] = log_prediction(
        result["dataset"], upload.file_ext, "macro2micro_pca_svc", result
    )
    return jsonify(result)


@app.post("/predict-pathology")
def predict_pathology():
    try:
        upload = validate_upload(request.files.get("image"))
    except ValidationError as error:
        return error_response(error.message, error.status_code)

    try:
        class_id, probability = model_loader.predict_tissue(upload.image)
    except model_loader.ModelNotFoundError:
        return error_response("PathMNIST AI загвар олдсонгүй.", 500)
    except Exception:
        app.logger.exception("/predict-pathology inference алдаа")
        return error_response("Pathology AI зураг боловсруулахад алдаа гарлаа.", 500)

    result = business_logic.interpret_tissue(class_id, probability)
    result["dataset"] = business_logic.DATASET_PATHMNIST
    result["analysis_id"] = log_prediction(
        result["dataset"], upload.file_ext, "pathmnist_cnn", result
    )
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=config.FLASK_DEBUG, port=config.PORT)
