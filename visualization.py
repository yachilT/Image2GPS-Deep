from typing import Optional
from architectures.dino_faiss import Match
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
import torch
import random
import numpy as np
import contextily as ctx
from pyproj import Transformer
from preprocess import GPSRectNorm
from private_utils import haversine_distance
import pandas as pd
import geopandas as gpd

transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857")

def to_mercator_coords(GPS_coords):
    xx, yy = transformer.transform(GPS_coords[:, 0], GPS_coords[:, 1])
    return np.stack([xx, yy], axis=1)

def _plot_map_panel(ax, gt_merc, pred_merc, err_m=None, nn_mercs=None, pad_m=50, zoom=19):
    all_x = [gt_merc[0], pred_merc[0]] + ([x for x, y in nn_mercs] if nn_mercs else [])
    all_y = [gt_merc[1], pred_merc[1]] + ([y for x, y in nn_mercs] if nn_mercs else [])

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    span_x = max_x - min_x
    span_y = max_y - min_y
    max_span = max(span_x, span_y)
    
    view_size = max_span + (2 * pad_m)
    half_size = view_size / 2

    ax.set_xlim(center_x - half_size, center_x + half_size)
    ax.set_ylim(center_y - half_size, center_y + half_size)
    ax.set_aspect('equal')

    try:
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=zoom)
    except Exception as e:
        print(f"Map fetch failed: {e}")

    ax.plot([gt_merc[0], pred_merc[0]], [gt_merc[1], pred_merc[1]], 
            color='red', linestyle='-', linewidth=2, alpha=0.7, zorder=4)

    # if nn_mercs:
    #     for i, (x, y) in enumerate(nn_mercs):
    #         ax.plot([gt_merc[0], x], [gt_merc[1], y], 
    #                 color='blue', linestyle='--', linewidth=0.8, alpha=0.6, zorder=3)

    ax.scatter(gt_merc[0], gt_merc[1], s=250, marker="*", 
               c='red', edgecolors='black', zorder=10)
    # ax.text(gt_merc[0], gt_merc[1], "GT", fontsize=11, fontweight='bold', zorder=12)

    ax.scatter(pred_merc[0], pred_merc[1], s=120, marker="s", 
               c='#39FF14', edgecolors='black', zorder=9)
    # ax.text(pred_merc[0], pred_merc[1], " Pred", fontsize=9, fontweight='bold', zorder=12)

    if nn_mercs:
        for i, (x, y) in enumerate(nn_mercs, 1):
            ax.scatter(x, y, s=100, marker="o", c='cyan', edgecolors='black', zorder=8, alpha=0.4)
            ax.text(x, y, f"NN{i}", fontsize=11, zorder=12)

    ax.legend(loc="best", fontsize=9, framealpha=0.8)
    if err_m is not None:
        ax.set_title(f"Error: {err_m:.2f} (m)", fontsize=18)
    ax.axis("off")

def visualize_model_predictions(model, dataset, gps_norm: GPSRectNorm, num_samples=12):
    IMG_MEAN=[0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]
    
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
    gt_batch = np.stack([gt.cpu().numpy() for gt in gt_list]) # (B, 2)
    pred_batch = None
    
    model_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        images_tensor = images_tensor.to(model_device)
        print("predicting...")
        pred_batch = model.predict_gps(images_tensor)

    if pred_batch is None:
        raise RuntimeError("Couldn't compute predictions")

    print("DONE!")   
    print("Preparing data for visualization...") 

    # getting GPS coords
    gt_GPS_batch = gps_norm.decode_np(gt_batch)
    pred_GPS_batch = gps_norm.decode_np(pred_batch)

    # convert to mercator projection
    gt_merc_b = to_mercator_coords(gt_GPS_batch)
    pred_merc_b = to_mercator_coords(pred_GPS_batch)

    # calculate meter error
    errors_m = haversine_distance(gt_GPS_batch, pred_GPS_batch)

    # moving images to from normalzied form
    mean = torch.tensor(IMG_MEAN).view(1, 3, 1, 1).to(model_device)
    std = torch.tensor(IMG_STD).view(1, 3, 1, 1).to(model_device)
    images_tensor = images_tensor * std + mean
    image_b = images_tensor.permute(0, 2, 3, 1).cpu().numpy()
    
    
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 16))  

    for k, (idx, image, gt_merc, pred_merc, err_m) in enumerate(tqdm(zip(indices, image_b, gt_merc_b, pred_merc_b, errors_m), total=num_samples)):
        # Calculate grid position
        row = k // SAMPLES_PER_ROW
        col_start = (k % SAMPLES_PER_ROW) * COLS_PER_SAMPLE
        
        ax_img = axes[row, col_start]
        ax_map = axes[row, col_start + 1]


        ax_img.imshow(image)
        ax_img.set_title(f"Sample {idx}", fontsize=18)
        ax_img.axis('off')

        _plot_map_panel(ax_map, gt_merc, pred_merc, err_m.item())

    # --- CLEANUP EMPTY SLOTS ---
    total_slots = n_rows * SAMPLES_PER_ROW
    for k in range(num_samples, total_slots):
        row = k // SAMPLES_PER_ROW
        col_start = (k % SAMPLES_PER_ROW) * COLS_PER_SAMPLE
        axes[row, col_start].axis('off')     # Hide unused image slot
        axes[row, col_start + 1].axis('off') # Hide unused map slot

    # Create custom handles for the legend
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', label='Ground Truth',
               markerfacecolor='red', markeredgecolor='black', markersize=15),
        Line2D([0], [0], marker='s', color='w', label='Prediction',
               markerfacecolor='#39FF14', markeredgecolor='black', markersize=10),
        Line2D([0], [0], color='red', lw=2, label='Error Line'),
    ]

    # Place legend at the top center of the entire Figure
    # bbox_to_anchor=(x, y) coordinates are relative to the figure
    fig.legend(handles=legend_elements, loc='upper center', 
               bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=18, frameon=False)
    plt.tight_layout()
    plt.show()

def plot_locations_from_csv(file_path):
    df = pd.read_csv(file_path)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']),
        crs="EPSG:4326"
    )
    gdf_mercator = gdf.to_crs(epsg=3857)
    fig, ax = plt.subplots(figsize=(12, 12))
    gdf_mercator.plot(ax=ax, color='red', markersize=50, alpha=0.8, edgecolor='k')
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    ax.set_axis_off()
    plt.show()

    
def _to_numpy_image(img, mean=None, std=None):
    if isinstance(img, torch.Tensor):
        x = img.detach().cpu()
        if x.ndim == 3 and x.shape[0] == 3:
            x = x.permute(1, 2, 0)  # HWC
        x = x.float().numpy()
    else:
        x = np.array(img)

    if mean is not None and std is not None:
        mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        x = x * std + mean

    if x.max() > 1.5:
        x = x / 255.0

    return np.clip(x, 0.0, 1.0).astype(np.float32)

# ----------------------------
# Main visualization
# ----------------------------
def plot_queries_with_topk_matches_and_map(
    db,
    test_dataset,
    GPS_norm,            
    train_dataset,
    n=5,
    k=3,
    weighted=True,
    mean=None, std=None,
    figsize_per_row=(16, 4),
    pad_m=50,
    zoom=19,
):
    if GPS_norm is None:
        raise ValueError("GPS_norm must be provided (object with .decode(lat_n, lon_n)->(lat_deg, lon_deg))")

    n = min(n, len(test_dataset))
    cols = k + 2
    fig = plt.figure(figsize=(figsize_per_row[0], figsize_per_row[1] * n))

    indices = random.sample(range(len(test_dataset)), n)

    for row, idx in enumerate(tqdm(indices)):
        query_img, gt_coords_n = test_dataset[idx]
        gt_coords_n = gt_coords_n.unsqueeze(0).cpu().numpy()

        orig_idx = test_dataset.indices[row] if hasattr(test_dataset, "indices") else row

        with torch.no_grad():
            pred_coords, matches = db.predict_gps(query_img, return_matches=True)

        pred_coords = GPS_norm.decode_np(pred_coords)
        gt_coords = GPS_norm.decode_np(gt_coords_n)

        pred_err_m = haversine_distance(pred_coords, gt_coords).item()

        ax = plt.subplot(n, cols, row * cols + 1)
        ax.imshow(_to_numpy_image(query_img, mean=mean, std=std))
        ax.axis("off")
        ax.set_title(
            f"QUERY split#{row}\norig#{orig_idx + 1}",
            fontsize=18
        )

        gt_merc = to_mercator_coords(gt_coords)
        pred_merc = to_mercator_coords(pred_coords)

        match_mercs = []
        match_dists = []
        num_matches = len(matches)

        if num_matches > 0:
            match_gps_n = np.array([m.gps for m in matches])
            match_gps = GPS_norm.decode_np(match_gps_n)
            
            match_dists = haversine_distance(gt_coords, match_gps).flatten()
            match_mercs = to_mercator_coords(match_gps)

        nn_xy = []

        for j in range(k):
            axm = plt.subplot(n, cols, row * cols + 1 + (j + 1))

            if j >= num_matches:
                axm.axis("off")
                axm.set_title(f"NN{j+1} (no match)", fontsize=9)
                continue

            m = matches[j]
            geo_m = match_dists[j]
            match_merc = match_mercs[j]
            
            nn_xy.append(match_merc)

            if train_dataset is not None and m.idx is not None:
                try:
                    mimg, _ = train_dataset[m.idx]
                    axm.imshow(_to_numpy_image(mimg, mean=mean, std=std))
                    axm.axis("off")
                except Exception:
                    axm.axis("off")
                    axm.text(0.5, 0.5, f"train_dataset[{m.idx}] failed", ha="center", va="center")
            else:
                axm.axis("off")
                axm.text(0.5, 0.5, "No train image", ha="center", va="center")

            score_str = f"{float(m.score):.4f}"
            geo_str = f"{geo_m:.1f} m"

            axm.set_title(
                f"NN{j+1} idx={m.idx}\n"
                f"emb_score={score_str}\n"
                f"distance={geo_str}",
                fontsize=15
            )

        ax_map = plt.subplot(n, cols, row * cols + cols)
        _plot_map_panel(ax_map, gt_merc.squeeze(0), pred_merc.squeeze(0), err_m=pred_err_m, nn_mercs=nn_xy)

        # Create custom handles for the legend
        legend_elements = [
            # Ground Truth (Red Star)
            Line2D([0], [0], marker='*', color='w', label='Ground Truth',
                markerfacecolor='red', markeredgecolor='black', markersize=15),
            
            # Prediction (Green Square)
            Line2D([0], [0], marker='s', color='w', label='Prediction',
                markerfacecolor='#39FF14', markeredgecolor='black', markersize=10),
            
            # Nearest Neighbors (Cyan Circle)
            Line2D([0], [0], marker='o', color='w', label='Nearest Neighbor',
                markerfacecolor='cyan', markeredgecolor='black', markersize=10),

            # Error Line (Red Solid)
            Line2D([0], [0], color='red', lw=2, linestyle='-', label='Pred Error'),

            # NN Connection Line (Blue Dashed)
            Line2D([0], [0], color='blue', lw=1, linestyle='--', label='GT-NN Dist'),
        ]

    # Place legend at the top center of the entire Figure
    # bbox_to_anchor=(x, y) coordinates are relative to the figure
    fig.legend(handles=legend_elements, loc='upper center', 
               bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=18, frameon=False)
    plt.tight_layout()
    plt.show()