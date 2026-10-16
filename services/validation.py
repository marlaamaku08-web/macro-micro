"""API layer-ийн upload validation."""

import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

import config

MSG_MISSING = "Зураг сонгоно уу."
MSG_BAD_EXTENSION = "Зөвхөн PNG, JPG, JPEG зураг оруулна уу."
MSG_INVALID_IMAGE = "Зургийг уншиж чадсангүй. PNG эсвэл JPG зураг сонгоно уу."
ALLOWED_FORMATS = {"PNG", "JPEG", "MPO"}


def too_large_message():
    return f"Зургийн хэмжээ {config.MAX_UPLOAD_MB:g} MB-аас ихгүй байх ёстой."


class ValidationError(Exception):
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass
class ValidatedUpload:
    filename: str
    file_ext: str
    image: Image.Image


def validate_upload(uploaded):
    """Upload хийсэн файлыг шалгаж, RGB PIL зураг буцаана.

    Алдаатай оролтод ValidationError (4xx) шиднэ.
    """
    if uploaded is None or not uploaded.filename or not uploaded.filename.strip():
        raise ValidationError(MSG_MISSING, 400)

    file_ext = Path(uploaded.filename).suffix.lower()
    if file_ext not in config.ALLOWED_EXTENSIONS:
        raise ValidationError(MSG_BAD_EXTENSION, 415)

    data = uploaded.read(config.MAX_UPLOAD_BYTES + 1)
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise ValidationError(too_large_message(), 413)
    if not data:
        raise ValidationError(MSG_INVALID_IMAGE, 400)

    try:
        with Image.open(io.BytesIO(data)) as probe:
            image_format = probe.format
            probe.verify()

        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise ValidationError(MSG_INVALID_IMAGE, 400)

    if image_format not in ALLOWED_FORMATS:
        raise ValidationError(MSG_BAD_EXTENSION, 415)

    return ValidatedUpload(
        filename=uploaded.filename,
        file_ext=file_ext,
        image=image.convert("RGB"),
    )
