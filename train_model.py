
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA

import numpy as np
import pickle


# Зургийн хавтас
BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "images"

# Зургийн хэмжээг ижил болгох
IMAGE_SIZE = (64, 64)

X = []
y = []

# Макро зураг
macro_folder = IMAGE_DIR / "Gross_images"

# Микроскоп зураг (хавтасны нэр диск дээрхтэй ижил — Linux дээр том/жижиг үсэг ялгана)
micro_folders = [
    IMAGE_DIR / "Histology_brain",
    IMAGE_DIR / "histology_heart",
]

# Pathology зураг биш тул сургалтаас хасна
# (Ukrainian_microscope_(cropped).jpg — микроскоп багажны гэрэл зураг, макро specimen биш)
EXCLUDED_FILES = {
    "Ukrainian_microscope_(cropped).jpg",
}


def load_images(folder, label):
    if not folder.exists():
        print(f"Хавтас олдсонгүй: {folder}")
        return

    for file in folder.iterdir():
        if file.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            continue

        if file.name in EXCLUDED_FILES:
            print(f"Алгаслаа (pathology зураг биш): {file.name}")
            continue

        try:
            image = Image.open(file).convert("RGB")
            image = image.resize(IMAGE_SIZE)

            pixels = np.array(image, dtype=np.float32) / 255.0
            X.append(pixels.flatten())
            y.append(label)

            print(f"Уншлаа: {file.name} -> {label}")

        except Exception as error:
            print(f"Зураг уншихад алдаа: {file.name}: {error}")


load_images(macro_folder, "macro")

for folder in micro_folders:
    load_images(folder, "micro")


# Өгөгдөл шалгах
print("\nНийт зураг:", len(X))
print("Макро зураг:", y.count("macro"))
print("Микроскоп зураг:", y.count("micro"))

if len(X) < 4 or len(set(y)) < 2:
    raise ValueError(
        "Сургалт хийхэд зураг хүрэлцэхгүй эсвэл хоёр ангилал олдсонгүй."
    )

print(
    "\nАНХААР: Цөөн зурагтай тул энэ бол "
    "сургалтын туршилтын загвар юм."
)

# Өгөгдлийг сургах ба шалгах хэсэгт хуваах
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Эхний энгийн AI ангилагч
model = make_pipeline(
    StandardScaler(),
    PCA(n_components=0.95),
    SVC(kernel="linear", probability=True, random_state=42)
)

model.fit(X_train, y_train)

# Загварын үр дүн
predictions = model.predict(X_test)

print("\nШалгалтын accuracy:", accuracy_score(y_test, predictions))
print("\nАнгиллын тайлан:")
print(classification_report(y_test, predictions, zero_division=0))

# Сурсан загварыг хадгалах
model_path = BASE_DIR / "macro2micro_model.pkl"

with open(model_path, "wb") as file:
    pickle.dump(model, file)

print(f"\nЗагвар хадгалагдлаа: {model_path}")