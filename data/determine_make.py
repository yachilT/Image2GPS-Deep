import os
from PIL import Image
from PIL.ExifTags import TAGS
import shutil

def get_camera_info(folder_path):
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg'))]
    
    print(f"{'Filename':<30} | {'Manufacturer':<15} | {'Model'}")
    print("-" * 65)

    for filename in files:
        file_path = os.path.join(folder_path, filename)
        make = ""
        try:
            with Image.open(file_path) as img:
                # Get the raw EXIF data
                exif_data = img.getexif()
                
                # Tag 271 is 'Make' (Manufacturer), Tag 272 is 'Model'
                make = exif_data.get(271, "Unknown")
                model = exif_data.get(272, "Unknown")
                
                print(f"{filename[:28]:<30} | {str(make):<15} | {model}")
                
        except Exception as e:
            print(f"Could not read {filename}: {e}")
        
        if make == "samsung":
            shutil.copy2(file_path, os.path.join(r"snapshots\18.12.25-Roi", filename))
            print(f"Copied {filename}")

if __name__ == "__main__":
    # Update this to your folder path
    target_folder = r"indexed_photos"
    get_camera_info(target_folder)