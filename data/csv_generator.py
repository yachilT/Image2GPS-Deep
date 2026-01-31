import pandas as pd

INPUT_CSV = "data/photo_locations.csv"
OUTPUT_CSV = "data/gt_corrupted.csv"

def main():
    # Read input CSV
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

    # Validate required columns
    required_cols = {"Index", "Latitude", "Longitude"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")

    # Clean and construct output
    df = df[["Index", "Latitude", "Longitude"]].dropna()
    df["Index"] = df["Index"].astype(str).str.strip()
    df["Latitude"] = df["Latitude"].astype(str).str.strip()
    df["Longitude"] = df["Longitude"].astype(str).str.strip()

    # Create image_name column
    df["image_name"] = df["Index"] + ".jpg"

    # Reorder columns exactly as requested
    df_out = df[["Index", "image_name", "Latitude", "Longitude"]]

    # Write output
    df_out.to_csv(OUTPUT_CSV, index=False)

    print(f"[OK] Wrote {len(df_out)} rows to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
