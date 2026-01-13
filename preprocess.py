import os
import csv
import json
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
from torchvision.transforms import v2
LAT_MIN, LAT_MAX = 31.26174, 31.2624
LON_MIN, LON_MAX = 34.80081, 34.80454

class CampusGPSDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None, gps_normalizer=None, clamp_labels=True):
        self.image_dir = image_dir
        self.transform = transform
        self.gps_normalizer = gps_normalizer
        self.clamp_labels = clamp_labels

        self.samples = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = row["Index"]
                lat = float(row["Latitude"])
                lon = float(row["Longitude"])

                img_path = None
                for ext in (".jpg", ".jpeg", ".JPG", ".JPEG"):
                    candidate = os.path.join(image_dir, f"{idx}{ext}")
                    if os.path.exists(candidate):
                        img_path = candidate
                        break
                if img_path is None:
                    raise FileNotFoundError(f"Image for index {idx} not found in {image_dir}")

                self.samples.append((img_path, lat, lon))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lat, lon = self.samples[idx]

        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # Normalize GPS using rectangle bounds
        lat_n, lon_n = self.gps_normalizer.encode(lat, lon)

        if self.clamp_labels:
            lat_n = float(np.clip(lat_n, 0.0, 1.0))
            lon_n = float(np.clip(lon_n, 0.0, 1.0))

        gps = torch.tensor([lat_n, lon_n], dtype=torch.float32)
        return img, gps


class GPSRectNorm:
    def __init__(self, lat_min=LAT_MIN, lat_max=LAT_MAX, lon_min=LON_MIN, lon_max=LON_MAX, eps=1e-12):
        self.lat_min = float(lat_min)
        self.lat_max = float(lat_max)
        self.lon_min = float(lon_min)
        self.lon_max = float(lon_max)
        self.lat_range = max(self.lat_max - self.lat_min, eps)
        self.lon_range = max(self.lon_max - self.lon_min, eps)

    def encode(self, lat, lon):
        lat_n = (float(lat) - self.lat_min) / self.lat_range
        lon_n = (float(lon) - self.lon_min) / self.lon_range
        return lat_n, lon_n

    def decode(self, lat_n, lon_n):
        lat = self.lat_min + float(lat_n) * self.lat_range
        lon = self.lon_min + float(lon_n) * self.lon_range
        return lat, lon
