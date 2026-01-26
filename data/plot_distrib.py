import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import numpy as np

import os

def plot_locations_from_csv(file_path):
    df = pd.read_csv(file_path)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']),
        crs="EPSG:4326"
    )
    gdf_mercator = gdf.to_crs(epsg=3857)
    
    df['Label'] = df['Original Name'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])

    fig, ax = plt.subplots(figsize=(12, 12))
    

    # --- PADDING LOGIC (Vertical Only) ---
    minx, miny, maxx, maxy = gdf_mercator.total_bounds
    
    # Only calculate height buffer (20%)
    buffer_y = (maxy - miny) * 0.2
    buffer_x = (maxx - minx) * 0.03
    # Set X limits strictly to the data bounds (No padding)
    ax.set_xlim(3874021.9633588814, 3874429.4245325592)
    
    # Set Y limits with padding
    ax.set_ylim(3666765.143721898, 3666877.2702055983)

    print("--- COPY THESE VALUES ---")
    print(f"XLIM = {ax.get_xlim()}")
    print(f"YLIM = {ax.get_ylim()}")
    print("-------------------------")
    
    # 6. Plot Points
    # gdf_mercator.plot(ax=ax, color='red', markersize=50, alpha=0.5, edgecolor='k')
    # gdf_mercator.plot(ax=ax, color=np.random.rand(len(gdf_mercator), 3), markersize=50, alpha=0.5, edgecolor='k')
    bright_colors = ['#FF00FF', '#32CD32'] 
    
    point_colors = [bright_colors[i % len(bright_colors)] for i in range(len(gdf_mercator))]

    gdf_mercator.plot(ax=ax, color=point_colors, markersize=50, alpha=0.5, edgecolor='k')

    # govmap_url = "https://cdnil.govmap.gov.il/xyz/ortho/{z}/{x}/{y}.png"
    # ctx.add_basemap(ax, source=govmap_url)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    ax.set_axis_off()
    plt.show()



def main():
    plot_locations_from_csv('photo_locations.csv')
    plot_locations_from_csv('corrected_photo_locations.csv')



if __name__ == '__main__':
    main()