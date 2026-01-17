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
import numpy as np

def haversine_distance(coords1: np.ndarray, coords2: np.ndarray):
    """
    Vectorized Haversine distance calculation.
    
    Args:
        coords1: (N, 2) numpy array of [Latitude, Longitude]
        coords2: (N, 2) numpy array of [Latitude, Longitude]
        
    Returns:
        (N, 1) numpy array of distances in meters
    """
    # Earth radius in meters
    R = 6371000.0

    # Convert degrees to radians
    # Shape becomes (N, 2)
    rads1 = np.radians(coords1)
    rads2 = np.radians(coords2)

    # Unpack columns: lat is index 0, lon is index 1
    lat1, lon1 = rads1[:, 0], rads1[:, 1]
    lat2, lon2 = rads2[:, 0], rads2[:, 1]

    # Differences
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = np.sin(dlat / 2)**2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    distance = R * c

    # Reshape to (N, 1) as requested
    return distance.reshape(-1, 1)



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
