# submission.py
# ============================================================
# End-to-end GPS prediction from a single RGB image
# ============================================================

import os
import shutil
import numpy as np
import torch
from private_utils import denorm_gps
import torchvision.transforms.v2 as v2

# ---------- YOUR IMPORT ----------
# adjust path if needed
from architectures.dino_faiss import SaladFaissGPSDB  

# ---------- Drive ----------
import gdown

DRIVE_MODELS_FOLDER_URL = "https://drive.google.com/drive/folders/18_ALZ-Xdz74LFBQaedxHROuL4Ex0qD-X"

ASSETS_DIR = "trained_models"
TMP_DL_DIR = "_tmp_drive_models_download"
FAISS_DB_NAME = "salad_faiss_db_norm3"
DB_PATH = os.path.join(ASSETS_DIR, FAISS_DB_NAME)


# ---------- Image preprocessing ----------
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

TRANSFORM = v2.Compose([
    v2.Resize((4004, 3010), interpolation=v2.InterpolationMode.BILINEAR),
    v2.Resize((686, 518), interpolation=v2.InterpolationMode.BILINEAR),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=IMG_MEAN, std=IMG_STD),
])

# ---------- Globals (lazy init) ----------
_DEVICE = None
_DINO_MODEL = None
_DB = None


# ============================================================
# Download from Drive
# ============================================================

def download_assets_if_needed():
    os.makedirs(ASSETS_DIR, exist_ok=True)

    db_ok = (
        os.path.isdir(DB_PATH)
        and os.path.isfile(os.path.join(DB_PATH, "index.faiss"))
        and os.path.isfile(os.path.join(DB_PATH, "data.json"))
    )

    if db_ok:
        return

    if os.path.exists(TMP_DL_DIR):
        shutil.rmtree(TMP_DL_DIR)

    gdown.download_folder(
        url=DRIVE_MODELS_FOLDER_URL,
        output=TMP_DL_DIR,
        quiet=False,
        use_cookies=False,
    )

    # copy FAISS DB folder
    for root, dirs, _ in os.walk(TMP_DL_DIR):
        if FAISS_DB_NAME in dirs:
            src = os.path.join(root, FAISS_DB_NAME)
            if os.path.exists(DB_PATH):
                shutil.rmtree(DB_PATH)
            shutil.copytree(src, DB_PATH)
            break

    shutil.rmtree(TMP_DL_DIR)


# ============================================================
# Initialization
# ============================================================

def _init_once():
    global _DEVICE, _DINO_MODEL, _DB

    if _DB is not None:
        return

    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) download assets
    download_assets_if_needed()

    # 2) load SALAD
    _DINO_MODEL = torch.hub.load(
        "serizba/salad",
        "dinov2_salad",
    ).eval().to(_DEVICE)

    # 3) load FAISS DB
    _DB = SaladFaissGPSDB.load(
        DB_PATH,
        model=_DINO_MODEL,
        device=_DEVICE,
    )


# ============================================================
# Required API
# ============================================================

def predict_gps(image: np.ndarray) -> np.ndarray:
    """
    Predict GPS latitude and longitude from a single RGB image.

    Input:
        image: np.ndarray (H, W, 3), uint8, RGB, [0,255]

    Output:
        np.ndarray (2,), float32
        [latitude, longitude]
    """
    _init_once()

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be numpy.ndarray")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3)")
    if image.dtype != np.uint8:
        raise ValueError("image dtype must be uint8")

    # numpy -> torch
    x = torch.from_numpy(image).permute(2, 0, 1)   # [3,H,W]
    x = TRANSFORM(x)
    x = x.unsqueeze(0).to(_DEVICE)                 # [1,3,H,W]

    # pred is normalized GPS: shape [1,2]
    pred_norm = _DB.predict_gps(x, weighted=True, return_numpy=True)  # float64/float32

    # decode normalized -> absolute lat/lon
    lat, lon = denorm_gps(pred_norm)  # each is shape [1] (or scalar-like)

    # pack into required output shape (2,) float32
    out = np.array([lat[0], lon[0]], dtype=np.float32)
    return out

# optional alias (some evaluators expect this exact name)
def predict_GPS(image: np.ndarray) -> np.ndarray:
    return predict_gps(image)
