import numpy as np


# --- PHYSICAL METEOROLOGICAL MODELING ---
def haversine(lat1, lon1, lat2, lon2):
    """Calculates the great-circle distance between two points in km."""
    R = 6371.0 # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def rankine_vortex(v_max, r, r_max=30, x=0.5):
    """Calculates local wind speed at distance r (km) using the Rankine decay model."""
    if r == 0: return 0
    if r <= r_max:
        return v_max * (r / r_max)
    else:
        return v_max * (r_max / r)**x

def vulnerability_curve(local_wind):
    """Fragility curve mapping local wind speed (m/s) to physical failure probability."""
    if local_wind < 15: # Below Tropical Storm threshold
        return 0.02
    elif local_wind < 30: # Tropical Storm to Cat 1
        return 0.02 + 0.04 * (local_wind - 15)
    else: # Cat 2+ Destructive Winds
        return min(0.95, 0.62 + 0.03 * (local_wind - 30))