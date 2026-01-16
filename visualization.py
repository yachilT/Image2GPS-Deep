from typing import Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import random
import numpy as np
import contextily as ctx
from pyproj import Transformer
from preprocess import GPSRectNorm

def _plot_map_panel(ax, gt_merc, pred_merc, nn_mercs=None, pad_m=50, zoom='auto'):
    # 1. Gather all points to find the center
    all_x = [gt_merc[0], pred_merc[0]] + ([x for x, y in nn_mercs] if nn_mercs else [])
    all_y = [gt_merc[1], pred_merc[1]] + ([y for x, y in nn_mercs] if nn_mercs else [])

    # 2. Calculate Center and Span
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    # Determine the largest span (width or height) to make it a square
    span_x = max_x - min_x
    span_y = max_y - min_y
    max_span = max(span_x, span_y)
    
    # Add padding to the largest span
    view_size = max_span + (2 * pad_m)
    half_size = view_size / 2

    # 3. Set Square Limits centered on the points
    ax.set_xlim(center_x - half_size, center_x + half_size)
    ax.set_ylim(center_y - half_size, center_y + half_size)
    
    # 4. FORCE EQUAL ASPECT RATIO (Crucial for maps)
    ax.set_aspect('equal')

    # 5. Add Background Map
    try:
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=zoom)
    except Exception as e:
        print(f"Map fetch failed: {e}")

    # --- DRAWING LOGIC (Same as before) ---
    ax.plot([gt_merc[0], pred_merc[0]], [gt_merc[1], pred_merc[1]], 
            color='red', linestyle='-', linewidth=2, alpha=0.7, zorder=4)

    if nn_mercs:
        for x, y in nn_mercs:
            ax.plot([gt_merc[0], x], [gt_merc[1], y], 
                    color='blue', linestyle='--', linewidth=0.8, alpha=0.6, zorder=3)

    ax.scatter(gt_merc[0], gt_merc[1], s=250, marker="*", 
               c='red', edgecolors='black', zorder=10, label='GT')
    ax.text(gt_merc[0], gt_merc[1], " GT", fontsize=11, fontweight='bold', zorder=12)

    ax.scatter(pred_merc[0], pred_merc[1], s=120, marker="s", 
               c='#39FF14', edgecolors='black', zorder=9, label='Pred')
    ax.text(pred_merc[0], pred_merc[1], " Pred", fontsize=9, fontweight='bold', zorder=12)

    if nn_mercs:
        for i, (x, y) in enumerate(nn_mercs, 1):
            ax.scatter(x, y, s=100, marker="o", 
                       c='cyan', edgecolors='black', zorder=8)
            ax.text(x, y, f" {i}", fontsize=8, zorder=12)

    ax.set_title(f"Spatial Distribution", fontsize=9)
    ax.axis("off")


def visualize_model_predictions(model, dataset, gps_norm: Optional[GPSRectNorm] = None):
    model.eval()
    
    # Transformer: Lat/Lon (4326) -> Web Mercator (3857)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    
    indices = random.sample(range(len(dataset)), 10)
    fig, axes = plt.subplots(10, 2, figsize=(8, 30))
    
    for i, idx in enumerate(tqdm(indices)):
        image, gt_coords = dataset[idx]
        
        # Prepare input
        if hasattr(image, 'unsqueeze'):
            input_tensor = image.unsqueeze(0)
        else:
            input_tensor = torch.tensor(image).unsqueeze(0)
            
        # Inference
        with torch.no_grad():
            pred_coords = model.predict_gps(input_tensor).squeeze(0).cpu().numpy()
            
        if isinstance(gt_coords, torch.Tensor):
            gt_coords = gt_coords.cpu().numpy()
        
        

        # Denormalize (Model output 0-1 -> Lat/Lon)
        if gps_norm:
            gt_lonlat = gps_norm.decode_np(gt_coords)
            pred_lonlat = gps_norm.decode_np(pred_coords)
        else:
            gt_lonlat = gt_coords
            pred_lonlat = pred_coords


        # Project (Lat/Lon -> Mercator Meters)
        # Note: Ensure input is (Longitude, Latitude) order for transformer
        gt_merc = transformer.transform(gt_lonlat[1], gt_lonlat[0])
        pred_merc = transformer.transform(pred_lonlat[1], pred_lonlat[0])

        # Plot Image (Left Panel)
        if isinstance(image, torch.Tensor):
            disp_image = image.permute(1, 2, 0).cpu().numpy()
            disp_image = (disp_image - disp_image.min()) / (disp_image.max() - disp_image.min())
        else:
            disp_image = image

        ax_img = axes[i, 0]
        ax_img.imshow(disp_image)
        ax_img.set_title(f"Input Image {idx}")
        ax_img.axis('off')

        # Plot Map (Right Panel)
        ax_map = axes[i, 1]
        _plot_map_panel(ax_map, gt_merc, pred_merc)

    plt.tight_layout()
    plt.show()