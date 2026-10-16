"""Model layer: сургасан model-уудыг process дотор нэг удаа ачаалж, дахин ашиглана.

Энд зөвхөн model ачаалах болон түүхий inference байна. Model output-ийг
шийдвэр болгох логик services/business_logic.py дотор байрлана.
"""

import pickle
import threading

import numpy as np

import config

_lock = threading.Lock()
_cache = {}


class ModelNotFoundError(Exception):
    """Model файл олдоогүй."""


def _load_once(key, path, loader):
    model = _cache.get(key)
    if model is not None:
        return model

    with _lock:
        model = _cache.get(key)
        if model is None:
            if not path.exists():
                raise ModelNotFoundError(str(path))
            model = loader(path)
            _cache[key] = model
    return model


def _load_pickle(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def _load_keras(path):
    # TensorFlow-г зөвхөн хэрэгтэй үед import хийнэ (server хурдан асна)
    import tensorflow as tf

    return tf.keras.models.load_model(path)


def get_macro2micro_model():
    return _load_once("macro2micro", config.MACRO2MICRO_MODEL_PATH, _load_pickle)


def get_pathmnist_model():
    return _load_once("pathmnist", config.PATHMNIST_MODEL_PATH, _load_keras)


def _to_pixels(image):
    image = image.convert("RGB").resize(config.IMAGE_SIZE)
    return np.array(image, dtype=np.float32) / 255.0


def predict_image_type(image):
    """Macro/micro ангилал. Буцаах утга: (class_name, probability 0–1)."""
    model = get_macro2micro_model()
    pixels = _to_pixels(image).flatten().reshape(1, -1)

    prediction = str(model.predict(pixels)[0])
    probabilities = model.predict_proba(pixels)[0]
    classes = [str(c) for c in model.classes_]
    return prediction, float(probabilities[classes.index(prediction)])


def predict_tissue(image):
    """PathMNIST 9 ангилал. Буцаах утга: (class_id, probability 0–1)."""
    model = get_pathmnist_model()
    pixels = np.expand_dims(_to_pixels(image), axis=0)

    probabilities = model.predict(pixels, verbose=0)[0]
    return int(np.argmax(probabilities)), float(np.max(probabilities))
