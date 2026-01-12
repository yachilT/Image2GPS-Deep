import os
from PIL import Image, ImageOps

def delete_horizontal_images(folder_path):
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg'))]
    
    deleted_count = 0
    remaining_count = 0

    for filename in files:
        file_path = os.path.join(folder_path, filename)
        
        try:
            with Image.open(file_path) as img:
                img = ImageOps.exif_transpose(img)
                width, height = img.size
                img.close()
                
                if width > height:
                    os.remove(file_path)
                    deleted_count += 1
                    print(f"Removed: {file_path}")
                else:
                    remaining_count += 1
        except Exception as e:
            print(f"Could not process {filename}: {e}")

    print(f"Cleanup complete.")
    print(f"Deleted (horizontal): {deleted_count}")
    print(f"Remaining (vertical/square): {remaining_count}")

if __name__ == "__main__":
    target_folder = "snapshots/18.12.25-Roi"
    delete_horizontal_images(target_folder)