from typing import Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import random
import numpy as np
import contextily as ctx
from pyproj import Transformer
from preprocess import GPSRectNorm


def _plot_map_panel(ax, gt_merc, pred_merc, nn_mercs=None, pad_m=50, zoom=19):
    # 1. Gather all points
    all_x = [gt_merc[0], pred_merc[0]] + ([x for x, y in nn_mercs] if nn_mercs else [])
    all_y = [gt_merc[1], pred_merc[1]] + ([y for x, y in nn_mercs] if nn_mercs else [])

    # 2. Calculate Center and Span for Square Aspect Ratio
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    span_x = max_x - min_x
    span_y = max_y - min_y
    max_span = max(span_x, span_y)
    
    view_size = max_span + (2 * pad_m)
    half_size = view_size / 2

    # 3. Set Limits & Aspect
    ax.set_xlim(center_x - half_size, center_x + half_size)
    ax.set_ylim(center_y - half_size, center_y + half_size)
    ax.set_aspect('equal')

    # 4. Add Background Map
    try:
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=zoom)
    except Exception as e:
        print(f"Map fetch failed: {e}")

    # --- DRAWING ---
    ax.plot([gt_merc[0], pred_merc[0]], [gt_merc[1], pred_merc[1]], 
            color='red', linestyle='-', linewidth=2, alpha=0.7, zorder=4)

    if nn_mercs:
        for x, y in nn_mercs:
            ax.plot([gt_merc[0], x], [gt_merc[1], y], 
                    color='blue', linestyle='--', linewidth=0.8, alpha=0.6, zorder=3)

    ax.scatter(gt_merc[0], gt_merc[1], s=250, marker="*", 
               c='red', edgecolors='black', zorder=10, label='GT')
    # Removed text to reduce clutter in smaller grid
    # ax.text(gt_merc[0], gt_merc[1], " GT", fontsize=11, fontweight='bold', zorder=12)

    ax.scatter(pred_merc[0], pred_merc[1], s=120, marker="s", 
               c='#39FF14', edgecolors='black', zorder=9, label='Pred')
    # ax.text(pred_merc[0], pred_merc[1], " Pred", fontsize=9, fontweight='bold', zorder=12)

    if nn_mercs:
        for i, (x, y) in enumerate(nn_mercs, 1):
            ax.scatter(x, y, s=100, marker="o", c='cyan', edgecolors='black', zorder=8)

    ax.axis("off")

def visualize_model_predictions(model, dataset, gps_norm: Optional[GPSRectNorm] = None, num_samples=12):
    IMG_MEAN=[0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]
    model.eval()
    
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    SAMPLES_PER_ROW = 3
    COLS_PER_SAMPLE = 2  # (Image, Map)
    
    # Calculate grid dimensions
    n_rows = int(np.ceil(num_samples / SAMPLES_PER_ROW))  # ceil(10/3) = 4
    n_cols = SAMPLES_PER_ROW * COLS_PER_SAMPLE          # 3 * 2 = 6
    
    print(f"Sampling {num_samples} samples from dataset")
    indices = random.sample(range(len(dataset)), num_samples)
    batch_data = [dataset[i] for i in indices]
    
    # Unzip into separate lists
    images_list, gt_list = zip(*batch_data)
    

    images_tensor = torch.stack(images_list) # (B, 3, H, W)
    gt_batch = np.stack([gt.cpu().numpy() for gt in gt_list])
    pred_batch = None

    device = next(model.parameters()).device

    with torch.no_grad():
        images_tensor = images_tensor.to(device)
        print("predicting...")
        pred_batch = model.predict_gps(images_tensor)
    print("DONE!")
    mean = torch.tensor(IMG_MEAN).view(1, 3, 1, 1).to(device)
    std = torch.tensor(IMG_STD).view(1, 3, 1, 1).to(device)
    images_tensor = images_tensor * std + mean
    image_batch = images_tensor.permute(0, 2, 3, 1).cpu().numpy()
    
    
    
    # Adjust figsize: Wider (20) because 6 columns, Shorter (16) because only 4 rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 16))
    
    # Flatten axes for easier iteration if desired, but here we index manually
    # axes shape is (4, 6)

    

    for k, (idx, image, gt_coords, pred_coords) in enumerate(tqdm(zip(indices, image_batch, gt_batch, pred_batch), total=num_samples)):
        # Calculate grid position
        row = k // SAMPLES_PER_ROW
        col_start = (k % SAMPLES_PER_ROW) * COLS_PER_SAMPLE
        
        ax_img = axes[row, col_start]
        ax_map = axes[row, col_start + 1]

        if gps_norm:
            gt_lonlat = gps_norm.decode_np(gt_coords)
            pred_lonlat = gps_norm.decode_np(pred_coords)
        else:
            gt_lonlat = gt_coords
            pred_lonlat = pred_coords

        gt_merc = transformer.transform(gt_lonlat[1], gt_lonlat[0])
        pred_merc = transformer.transform(pred_lonlat[1], pred_lonlat[0])

        ax_img.imshow(image)
        ax_img.set_title(f"Sample {idx}", fontsize=15)
        ax_img.axis('off')

        # --- PLOT MAP ---
        _plot_map_panel(ax_map, gt_merc, pred_merc)

    # --- CLEANUP EMPTY SLOTS ---
    # We have 10 samples, but the grid has space for 12 (4 rows * 3 samples).
    # Turn off the axes for the remaining empty slots in the last row.
    total_slots = n_rows * SAMPLES_PER_ROW
    for k in range(num_samples, total_slots):
        row = k // SAMPLES_PER_ROW
        col_start = (k % SAMPLES_PER_ROW) * COLS_PER_SAMPLE
        axes[row, col_start].axis('off')     # Hide unused image slot
        axes[row, col_start + 1].axis('off') # Hide unused map slot

    plt.tight_layout()
    plt.show()