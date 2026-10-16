"""Business logic layer: model output-ийг application шийдвэр болгон хувиргана.

- detect_dataset: файлын нэрээр dataset тодорхойлох
- interpret_image_type: macro/micro model output → шийдвэр
- interpret_tissue: PathMNIST 9 ангиллын output → шийдвэр + тайлбар

Confidence threshold-оос бага үед decision = "low_confidence",
бусад үед decision = "classified".
"""

import config

DATASET_PATHMNIST = "PathMNIST"
DATASET_TOXOPATH = "ToxoPath-1"

DECISION_CLASSIFIED = "classified"
DECISION_LOW_CONFIDENCE = "low_confidence"

LOW_CONFIDENCE_MESSAGE = (
    "Итгэлцлийн түвшин бага. Үр дүнг сургалтын хүрээнд дахин нягтална уу."
)

IMAGE_TYPE_NOTICE = (
    "Сургалтын туршилтын үр дүн. "
    "Эмнэлзүйн оношилгоонд ашиглахгүй."
)
TISSUE_NOTICE = (
    "PathMNIST-д суурилсан сургалтын AI. Эмнэлзүйн оношилгоонд ашиглахгүй."
)

IMAGE_TYPES = {
    "macro": {
        "label": "Макроскопийн зураг",
        "explanation": (
            "Макроскопийн зураг нь эрхтэн, эдийн өөрчлөлтийг нүдэн харааны "
            "түвшинд харуулна: голомтын байрлал, хэмжээ, өнгө, хил хязгаар."
        ),
    },
    "micro": {
        "label": "Микроскопийн зураг",
        "explanation": (
            "Микроскопийн зураг нь гистологийн бэлдмэлийг өсгөлттэй харуулж, "
            "эс болон эдийн бүтцийн түвшний өөрчлөлтийг ажиглах боломж олгоно."
        ),
    },
}

# PathMNIST 9 ангилал (index.html-ийн pathmnistInfo-тэй ижил агуулга)
PATHMNIST_CLASSES = {
    0: {
        "label": "Adipose",
        "morphology": "Том, бөөрөнхий эсвэл олон талт, цайвар хоосон харагдах өөхний эсүүд нимгэн мембранаар хүрээлэгдэн зөгийн сархинаг шиг бүтэц үүсгэнэ. Цөм нь жижиг, хавтгай бөгөөд эсийн захад шахагдсан байна.",
        "explanation": "Adipose буюу өөхний эд. Эсийн доторх липид бэлдмэл боловсруулах явцад уусдаг тул эсүүд хоосон, цагаан харагдана.",
    },
    1: {
        "label": "Background",
        "morphology": "Эдийн бүтэц, эс, цөм бараг харагдахгүй; цагаан, саарал эсвэл жигд өнгөтэй хоосон талбай давамгайлна.",
        "explanation": "Background буюу шилэн бэлдмэлийн эдгүй хэсэг. Оношилгооны мэдээлэл агуулахгүй боловч AI загварт эд ба эдгүй хэсгийг ялгахад хэрэгтэй ангилал юм.",
    },
    2: {
        "label": "Debris",
        "morphology": "Бүтэц нь алдагдсан, хэлбэргүй эозинофил масс, цөмийн хэлтэрхий (karyorrhexis), үхжсэн эсийн үлдэгдэл болон заримдаа цус алдалтын хэсгүүд ажиглагдана.",
        "explanation": "Debris буюу үхжлийн хог хаягдал. Ихэвчлэн хавдрын үхжил (tumor necrosis)-тэй холбоотой бөгөөд хурдан ургаж буй хавдарт цусан хангамж хүрэлцэхгүй үед үүснэ.",
    },
    3: {
        "label": "Lymphocytes",
        "morphology": "Жижиг, бөөрөнхий, нягт хар хөх (hyperchromatic) цөмтэй, цитоплазм нь маш бага эсүүд нягт бөөгнөрсөн байна.",
        "explanation": "Lymphocytes буюу лимфоцитын бөөгнөрөл. Хавдрын эсрэг дархлааны хариу урвал (tumor-infiltrating lymphocytes) эсвэл лимфоид эдийг илэрхийлж болно.",
    },
    4: {
        "label": "Mucus",
        "morphology": "Цайвар хөх эсвэл ягаавтар, бүдэг, эсгүй (acellular) нойтон мэт масс; эсийн бүтэц бараг харагдахгүй, заримдаа хөвж буй цөөн эс ажиглагдана.",
        "explanation": "Mucus буюу салс. Гэдэсний бокал эсүүдийн ялгаруулдаг бодис бөгөөд mucinous adenocarcinoma-д их хэмжээгээр хуримтлагдаж болно.",
    },
    5: {
        "label": "Smooth muscle",
        "morphology": "Урт, ээрүүл хэлбэрийн эозинофил цитоплазмтай эсүүд багцалж, нэг чиглэлд эгнэн байрлана. Цөм нь урт, мохоо үзүүртэй (cigar-shaped) хэлбэртэй.",
        "explanation": "Smooth muscle буюу гөлгөр булчин. Бүдүүн гэдэсний muscularis propria давхаргыг төлөөлөх ба хавдрын нэвчилтийн гүнийг үнэлэхэд чухал.",
    },
    6: {
        "label": "Normal colon mucosa",
        "morphology": "Жигд, зэрэгцээ байрласан шулуун хоолой хэлбэрийн булчирхайнууд (crypts), бокал эсүүд элбэг, цөм нь суурь мембраны дагуу жигд эгнэсэн, атипи ажиглагдахгүй.",
        "explanation": "Normal colon mucosa буюу бүдүүн гэдэсний хэвийн салст бүрхүүл. Хавдрын эдтэй харьцуулах хэвийн жишээ болно.",
    },
    7: {
        "label": "Cancer-associated stroma",
        "morphology": "Хавдрын булчирхайнуудын хооронд байрлах ээрүүл хэлбэрийн фибробласт, миофибробластууд болон эмх цэгцгүй, нягт коллаген ширхэгүүд (desmoplasia) ажиглагдана.",
        "explanation": "Cancer-associated stroma буюу хавдартай холбоотой суурь эд. Хавдрын эсүүдийн нөлөөгөөр холбогч эд урвалын өөрчлөлтөд орж, хавдрын өсөлт болон тархалтад оролцдог.",
    },
    8: {
        "label": "Colorectal adenocarcinoma epithelium",
        "morphology": "Хэвийн булчирхайн архитектур алдагдаж, атипик хавдрын эпителийн эсүүд нягт байрласан бүтэц ажиглагдаж болно.",
        "explanation": "Энэ ангилал нь PathMNIST dataset дахь colorectal adenocarcinoma epithelium буюу бүдүүн гэдэсний аденокарциномын эпителийн эдийг илэрхийлнэ.",
    },
}


def detect_dataset(filename):
    """"pathmnist_" гэж эхэлсэн файлыг PathMNIST, бусдыг ToxoPath-1 гэж үзнэ."""
    name = (filename or "").replace("\\", "/").rsplit("/", 1)[-1].lower()
    return DATASET_PATHMNIST if name.startswith("pathmnist_") else DATASET_TOXOPATH


def decide(probability, threshold=None):
    """Магадлал (0–1)-ыг threshold-той харьцуулж шийдвэр гаргана."""
    if threshold is None:
        threshold = config.CONFIDENCE_THRESHOLD
    if probability < threshold:
        return DECISION_LOW_CONFIDENCE, LOW_CONFIDENCE_MESSAGE
    return DECISION_CLASSIFIED, None


def _to_percent(probability):
    return round(float(probability) * 100, 2)


def interpret_image_type(prediction, probability):
    """Macro/micro model output → API response-ийн бизнес талбарууд."""
    if prediction not in IMAGE_TYPES:
        raise ValueError(f"Unknown image type: {prediction!r}")

    decision, decision_message = decide(probability)
    info = IMAGE_TYPES[prediction]
    return {
        "label": info["label"],
        "class": prediction,
        "confidence": _to_percent(probability),
        "notice": IMAGE_TYPE_NOTICE,
        "decision": decision,
        "decision_message": decision_message,
        "explanation": info["explanation"],
    }


def interpret_tissue(class_id, probability):
    """PathMNIST model output → API response-ийн бизнес талбарууд."""
    if class_id not in PATHMNIST_CLASSES:
        raise ValueError(f"Unknown PathMNIST class: {class_id!r}")

    decision, decision_message = decide(probability)
    info = PATHMNIST_CLASSES[class_id]
    return {
        "class": class_id,
        "label": info["label"],
        "confidence": _to_percent(probability),
        "notice": TISSUE_NOTICE,
        "decision": decision,
        "decision_message": decision_message,
        "morphology": info["morphology"],
        "explanation": info["explanation"],
    }
