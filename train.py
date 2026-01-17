import torch
from Retrival_Dino_Salad.model import SaladFaissGPSDB
from Retrival_Dino_Salad.dinomlp import GPSPredictor, train_dinomlp, get_dino_mlp
from preprocess import CampusGPSDataset, GPSRectNorm, get_dataset
import torchvision.transforms.v2 as v2
from private_utils import *
from tqdm import tqdm
from torch.utils.data import DataLoader

from typing import Optional
import numpy as np
from tqdm import tqdm
from private_utils import haversine_distance


def evaluate_gps_meters_batched(
    model,
    dataloader: DataLoader,
    GPS_norm: Optional[GPSRectNorm]=None,
    device=None,
):

    errors_list = []
    pbar = tqdm(dataloader, leave=False)
    for batch_idx, (images, gps_tensor) in enumerate(tqdm(pbar)):
        if device is not None:
            images = images.to(device)
        gt_gps = gps_tensor.cpu().numpy()

        preds = model.predict_gps(images)
        lat_std = preds[:, 0].std().item()
        lon_std = preds[:, 1].std().item()
        
        # Update the progress bar with the spread
        pbar.set_postfix({
            "std_lat": f"{lat_std:.4f}",  # Should NOT be 0.0000
            "std_lon": f"{lon_std:.4f}"   # Should NOT be 0.0000
        })

        # STOP EARLY if true collapse is detected
        if lat_std < 1e-4 and lon_std < 1e-4 and batch_idx > 5:
            tqdm.write(f"!! WARNING: Mode Collapse Detected at Batch {batch_idx} !!")
            tqdm.write(f"First 3 preds: \n{preds[:3].detach().cpu().numpy()}")

        preds_decoded = GPS_norm.decode_np(preds)
        gt_decoded = GPS_norm.decode_np(gt_gps)
        # print(f"predicted GPS coords: {preds_decoded}")

        batch_errs = haversine_distance(preds_decoded, gt_decoded)
        errors_list.append(batch_errs)

    errors_m = np.concatenate(errors_list, axis=0)

    thresholds = [10, 25, 50, 100, 250, 500, 1000]
    metrics = {
        "mean_m": float(errors_m.mean()),
        "median_m": float(np.median(errors_m)),
        "p90_m": float(np.percentile(errors_m, 90)),
        "p95_m": float(np.percentile(errors_m, 95)),
        "max_m": float(errors_m.max()),
    }

    for t in thresholds:
        metrics[f"within_{t}m_%"] = float((errors_m <= t).mean() * 100.0)

    return metrics, errors_m

device = "cuda" if torch.cuda.is_available() else "cpu"

DINO_DIR_PATH = "Retrival_Dino_Salad"






def main():
    gps_norm, full_dataset, train_dataset, val_dataset, train_loader, val_loader = get_dataset()
    dino_mlp = get_dino_mlp('trained_models/mlp_head3.pth', train_loader, val_loader)



if __name__ == '__main__':
    main()