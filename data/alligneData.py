import os
import cv2
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer
import contextily as ctx
from joblib import Memory

# 1. Setup Caching for map tiles
memory = Memory("map_cache", verbose=0)
cached_add_basemap = memory.cache(ctx.add_basemap)

# 2. Setup Coordinate Transformer (GPS <-> Map)
to_gps = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

CSV_FILE = "corrected_photo_locations.csv"  # Change this to your CSV filename
IMAGES_DIR = "indexed_photos"
LOG_FILE = "processing_progress.txt"

def load_progress():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            return set(line.strip() for line in f)
    return set()

def log_progress(index):
    with open(LOG_FILE, "a") as f:
        f.write(f"{index}\n")

def find_image_file(index, directory):
    """Finds image file named {index}.jpg, .jpeg, or .png"""
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        path = Path(directory) / f"{index}{ext}"
        if path.exists():
            return path
    return None

def manual_csv_gps_fixer():
    # Load CSV
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found!")
        return
    
    df = pd.read_csv(CSV_FILE)
    processed_indices = load_progress()

    print(f"Total rows in CSV: {len(df)}")
    start_input = input(f"Start from which INDEX (not row number)? ")
    try:
        current_start = int(start_input) if start_input.strip() != "" else df['Index'].min()
    except ValueError:
        current_start = df['Index'].min()

    # Setup Plotting
    plt.ion()
    fig, (ax_img, ax_map) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Filter dataframe to start from user's index
    working_df = df[df['Index'] >= current_start].copy()

    for idx, row in working_df.iterrows():
        img_idx = row['Index']
        
        if str(img_idx) in processed_indices:
            continue

        img_path = find_image_file(img_idx, IMAGES_DIR)
        
        # Clear UI
        ax_img.clear()
        ax_map.clear()

        if img_path is None:
            print(f"Skipping Index {img_idx}: Image file not found in {IMAGES_DIR}")
            continue

        # 1. Show Image
        img_raw = cv2.imread(str(img_path))
        if img_raw is not None:
            ax_img.imshow(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB))
        ax_img.set_title(f"Index: {img_idx}\nFile: {img_path.name}")
        ax_img.axis('off')

        # 2. Show Map (based on CSV original coords)
        orig_lat = row['Latitude']
        orig_lon = row['Longitude']
        
        if not pd.isna(orig_lat) and not pd.isna(orig_lon):
            mx, my = to_merc.transform(orig_lon, orig_lat)
            ax_map.set_xlim(mx - 150, mx + 150) # Tight zoom
            ax_map.set_ylim(my - 150, my + 150)
            try:
                ctx.add_basemap(ax_map, source=ctx.providers.OpenStreetMap.Mapnik, zoom=19)
                ax_map.scatter(mx, my, s=300, marker="X", c='red', edgecolors='white', label='Original')
            except:
                ax_map.set_title("Map Load Failed")
        
        ax_map.axis('off')
        plt.draw()
        plt.pause(0.01)

        # 3. Handle User Input
        print(f"\n--- Index {img_idx} ---")
        print(f"Original: {orig_lat}, {orig_lon}")
        val = input("Paste 'New Lat, New Lon' (s: skip, q: quit): ").strip()

        if val.lower() == 'q': 
            break
        if val.lower() == 's' or ',' not in val:
            print("Skipped.")
            continue

        try:
            new_lat, new_lon = map(float, val.split(','))
            
            # Update Dataframe in memory
            df.loc[df['Index'] == img_idx, 'Latitude'] = new_lat
            df.loc[df['Index'] == img_idx, 'Longitude'] = new_lon
            
            # Save the whole CSV after every update (safest for parallel work)
            df.to_csv(CSV_FILE, index=False)
            
            log_progress(img_idx)
            print(f"✅ Updated CSV for Index {img_idx}")
        except Exception as e:
            print(f"Input Error: {e}")

    plt.close()
    print("Session finished.")

if __name__ == "__main__":
    manual_csv_gps_fixer()