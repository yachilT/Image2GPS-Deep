import os
import csv
import shutil
import re
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def get_decimal_from_dms(dms, ref):
    degrees = dms[0]
    minutes = dms[1] / 60.0
    seconds = dms[2] / 3600.0
    if ref in ['S', 'W']:
        return -float(degrees + minutes + seconds)
    return float(degrees + minutes + seconds)

def extract_gps(image_path):
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        if not exif_data:
            return None

        gps_info = {}
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == "GPSInfo":
                for t in value:
                    sub_tag = GPSTAGS.get(t, t)
                    gps_info[sub_tag] = value[t]

        if 'GPSLatitude' in gps_info and 'GPSLongitude' in gps_info:
            lat = get_decimal_from_dms(gps_info['GPSLatitude'], gps_info['GPSLatitudeRef'])
            lon = get_decimal_from_dms(gps_info['GPSLongitude'], gps_info['GPSLongitudeRef'])
            return lat, lon
    except Exception as e:
        print(f"Error extracting GPS from {image_path}: {e}")
    return None

def get_highest_index(target_folder):
    """Finds the highest numbered filename in the target folder."""
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        return 0
    
    indices = [0]
    for filename in os.listdir(target_folder):
        # Matches numbers at the start of the filename before the extension
        match = re.match(r'^(\d+)\.', filename)
        if match:
            indices.append(int(match.group(1)))
    return max(indices)

def process_folder(source_folder, target_folder, output_csv):
    # 1. Get starting index
    current_index = get_highest_index(target_folder) + 1
    
    # 2. Check if CSV exists to decide on writing header
    file_exists = os.path.isfile(output_csv)

    # 3. Open CSV in Append mode ('a')
    with open(output_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Index', 'Original Name', 'Latitude', 'Longitude', 'Google Maps Link'])

        for filename in os.listdir(source_folder):
            if filename.lower().endswith(('.jpg', '.jpeg')):
                source_path = os.path.join(source_folder, filename)
                coords = extract_gps(source_path)
                
                if coords:
                    lat, lon = coords
                    maps_link = f"https://www.google.com/maps?q={lat},{lon}"
                    
                    # Define new filename and path
                    extension = os.path.splitext(filename)[1]
                    new_filename = f"{current_index}{extension}"
                    target_path = os.path.join(target_folder, new_filename)
                    
                    # Copy and Rename
                    shutil.copy2(source_path, target_path)
                    
                    # Write to CSV
                    writer.writerow([current_index, filename, lat, lon, maps_link])
                    
                    print(f"Saved: {filename} -> {new_filename}")
                    current_index += 1
                else:
                    print(f"Skipped (No GPS): {filename}")

# --- SETTINGS ---
src = "snapshots/11.1.26-Roi" 
dst = "./indexed_photos"
csv_name = "photo_locations.csv"

process_folder(src, dst, csv_name)
print(f"\nProcessing complete. Photos moved to '{dst}' and logged in '{csv_name}'")