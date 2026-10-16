
from pathlib import Path
import pickle

import numpy as np
from PIL import Image


# Төслийн хавтас
BASE_DIR = Path(__file__).resolve().parent

# Сурсан AI загвар
MODEL_PATH = BASE_DIR / "macro2micro_model.pkl"

# Зураг ижил хэмжээтэй байх ёстой
IMAGE_SIZE = (64, 64)


def predict_image(image_path):
    # Загвар байгаа эсэх
    if not MODEL_PATH.exists():
        print("AI загвар олдсонгүй. Эхлээд train_model.py ажиллуулаарай.")
        return

    # AI загварыг нээх
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    # Зургийг унших
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    # Зургийг тоон өгөгдөл болгох
    pixels = np.array(image, dtype=np.float32) / 255.0
    pixels = pixels.flatten().reshape(1, -1)

    # AI-аар ангилах
    prediction = model.predict(pixels)[0]
    probabilities = model.predict_proba(pixels)[0]

    # Ангиллын итгэлцлийн оноо
    classes = model.classes_
    confidence = probabilities[list(classes).index(prediction)] * 100

    print("\n--- Macro&Micro AI үр дүн ---")

    if prediction == "macro":
        print("Ангилал: Макроскопийн зураг")
    else:
        print("Ангилал: Микроскопийн зураг")

    print(f"Загварын итгэлцлийн оноо: {confidence:.2f}%")
    print("Анхааруулга: Энэ бол сургалтын туршилтын үр дүн.")


# Турших зураг
test_image = BASE_DIR / "images" / "Gross_images" / "Brain_001.png"

predict_image(test_image)