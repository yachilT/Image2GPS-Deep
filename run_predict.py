# run_predict.py
# ------------------------------------------------------------
# Example script that uses submission.predict_gps
# ------------------------------------------------------------

import numpy as np
from PIL import Image

# import YOUR submission
import submission


def load_image(path: str) -> np.ndarray:
    """
    Load image as RGB uint8 numpy array (H,W,3)
    """
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


if __name__ == "__main__":
    # path to a test image from the dataset
    IMAGE_PATH = "data/indexed_photos/1.jpg"

    image = load_image(IMAGE_PATH)

    print("Image shape:", image.shape, image.dtype)

    # ---- GPS prediction ----
    gps = submission.predict_gps(image)

    print("Predicted GPS:")
    print("  Latitude :", float(gps[0]))
    print("  Longitude:", float(gps[1]))
