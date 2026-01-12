import os
import csv
from typing import Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
from torchvision.transforms import v2

class CampusGPSDataset(Dataset):
    """
    Loads:
      - images from indexed_photos/
      - GPS labels from photo_locations.csv

    CSV format:
      Index, Original Name, Latitude, Longitude, Google Maps Link
    """

    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        transform=None,
    ):
        self.image_dir = image_dir
        self.transform = transform

        self.samples = []  # (image_path, lat, lon)

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = row["Index"]
                lat = float(row["Latitude"])
                lon = float(row["Longitude"])

                # image name is index + extension (jpg/jpeg)
                # try both, robustly
                img_path = None
                for ext in (".jpg", ".jpeg", ".JPG", ".JPEG"):
                    candidate = os.path.join(image_dir, f"{idx}{ext}")
                    if os.path.exists(candidate):
                        img_path = candidate
                        break

                if img_path is None:
                    raise FileNotFoundError(f"Image for index {idx} not found in {image_dir}")

                self.samples.append((img_path, lat, lon))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, lat, lon = self.samples[idx]

        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img).convert("RGB")
        if self.transform:
            img = self.transform(img)
            
        transform_to_tensor = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])
        img = transform_to_tensor(img)
        gps = torch.tensor([lat, lon], dtype=torch.float32)
        return img, gps
