import csv

INPUT_CSV = "data/corrected_photo_locations.csv"
OUTPUT_TXT = "data/gt.csv"  # change if you want

def main():
    rows_out = []

    with open(INPUT_CSV, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"Index", "Latitude", "Longitude"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"gt.csv missing columns: {missing}. Found: {reader.fieldnames}")

        for r in reader:
            idx = str(r["Index"]).strip()
            lat = str(r["Latitude"]).strip()
            lon = str(r["Longitude"]).strip()

            if not idx or not lat or not lon:
                continue

            image_name = f"{idx}.jpg"
            rows_out.append(f"{image_name} {lat} {lon}")

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(rows_out) + ("\n" if rows_out else ""))

    print(f"[OK] Wrote {len(rows_out)} lines to {OUTPUT_TXT}")

if __name__ == "__main__":
    main()
