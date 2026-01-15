# utils.py
import numpy as np

# -------------------------------------------------
# Grid / normalization constants
# -------------------------------------------------
LAT_MIN, LAT_MAX = 31.26174, 31.2624
LON_MIN, LON_MAX = 34.80081, 34.80454


# -------------------------------------------------
# Normalization helpers
# -------------------------------------------------
def norm_gps(lat, lon):
    """
    Convert lat/lon -> normalized [0,1] coords
    """
    lat_n = (lat - LAT_MIN) / (LAT_MAX - LAT_MIN)
    lon_n = (lon - LON_MIN) / (LON_MAX - LON_MIN)
    return lat_n, lon_n


def denorm_gps(gps_norm):
    """
    Convert normalized coords -> lat/lon

    gps_norm: (..., 2) array-like
    returns: (..., 2) lat/lon
    """
    gps_norm = np.asarray(gps_norm, dtype=np.float64)
    lat = LAT_MIN + gps_norm[..., 0] * (LAT_MAX - LAT_MIN)
    lon = LON_MIN + gps_norm[..., 1] * (LON_MAX - LON_MIN)
    return lat, lon


# -------------------------------------------------
# Geographic distance (meters)
# -------------------------------------------------
def haversine_m(lat1, lon1, lat2, lon2):
    """
    Great-circle distance in meters
    """
    lat1, lon1, lat2, lon2 = map(
        np.deg2rad, [lat1, lon1, lat2, lon2]
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return 6371000.0 * c


# -------------------------------------------------
# Error computation
# -------------------------------------------------
def gps_error_m(pred_norm, gt_norm):
    """
    Compute error (meters) between normalized GPS points
    """
    pred_lat, pred_lon = denorm_gps(pred_norm)
    gt_lat, gt_lon = denorm_gps(gt_norm)
    return haversine_m(pred_lat, pred_lon, gt_lat, gt_lon)


# -------------------------------------------------
# Metrics
# -------------------------------------------------
def summarize_errors(errors):
    """
    errors: 1D array of distances in meters
    """
    errors = np.asarray(errors, dtype=np.float64)

    return {
        "N": int(errors.size),
        "mean_m": float(errors.mean()),
        "median_m": float(np.median(errors)),
        "p75_m": float(np.percentile(errors, 75)),
        "p90_m": float(np.percentile(errors, 90)),
        "p95_m": float(np.percentile(errors, 95)),
        "acc@25m": float((errors <= 25).mean()),
        "acc@50m": float((errors <= 50).mean()),
        "acc@100m": float((errors <= 100).mean()),
        "acc@250m": float((errors <= 250).mean()),
        "acc@1km": float((errors <= 1000).mean()),
    }


# -------------------------------------------------
# Confidence (optional)
# -------------------------------------------------
def gps_weight_confidence(weights):
    """
    Simple confidence score from neighbor weights
    """
    w = np.asarray(weights, dtype=np.float64)
    return float(w.max())


# -------------------------------------------------
# Sanity checks
# -------------------------------------------------
def sanity_check():
    """
    Quick check that normalization is correct
    """
    a = denorm_gps([0.0, 0.0])
    b = denorm_gps([1.0, 1.0])
    return {
        "min_corner": a,
        "max_corner": b,
    }
