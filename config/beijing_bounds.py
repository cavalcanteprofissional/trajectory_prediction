# config/beijing_bounds.py
"""
Limites geográficos de Beijing para validação e clamp de predições

Beijing typical bounds:
- Latitude: 39.4°N to 40.5°N
- Longitude: 116.0°E to 117.0°E
"""

from typing import Tuple
import numpy as np

# Limites oficiales de Beijing (margem de segurança)
BEIJING_BOUNDS = {
    'lat_min': 39.4,
    'lat_max': 40.5,
    'lon_min': 116.0,
    'lon_max': 117.0
}

# Margem de segurança (para predições)
SAFETY_MARGIN = 0.05

BEIJING_BOUNDS_SAFE = {
    'lat_min': BEIJING_BOUNDS['lat_min'] - SAFETY_MARGIN,
    'lat_max': BEIJING_BOUNDS['lat_max'] + SAFETY_MARGIN,
    'lon_min': BEIJING_BOUNDS['lon_min'] - SAFETY_MARGIN,
    'lon_max': BEIJING_BOUNDS['lon_max'] + SAFETY_MARGIN
}

# Centro de Beijing
BEIJING_CENTER = {
    'lat': 39.9042,
    'lon': 116.4074
}


def clamp_to_beijing(lat: float, lon: float) -> Tuple[float, float]:
    """
    Limita coordenadas para dentro dos limites de Beijing.
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        tuple: (lat, lon) limitados
    """
    lat_clamped = np.clip(lat, BEIJING_BOUNDS['lat_min'], BEIJING_BOUNDS['lat_max'])
    lon_clamped = np.clip(lon, BEIJING_BOUNDS['lon_min'], BEIJING_BOUNDS['lon_max'])
    return float(lat_clamped), float(lon_clamped)


def clamp_to_beijing_array(
    lat_array: np.ndarray, 
    lon_array: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Limita arrays de coordenadas para dentro dos limites de Beijing.
    
    Args:
        lat_array: Array de latitudes
        lon_array: Array de longitudes
        
    Returns:
        tuple: (lat_array, lon_array) limitados
    """
    lat_clamped = np.clip(lat_array, BEIJING_BOUNDS['lat_min'], BEIJING_BOUNDS['lat_max'])
    lon_clamped = np.clip(lon_array, BEIJING_BOUNDS['lon_min'], BEIJING_BOUNDS['lon_max'])
    return lat_clamped, lon_clamped


def is_in_beijing(lat: float, lon: float) -> bool:
    """
    Verifica se coordenadas estão dentro de Beijing.
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        bool: True se estiver em Beijing
    """
    return (BEIJING_BOUNDS['lat_min'] <= lat <= BEIJING_BOUNDS['lat_max'] and
            BEIJING_BOUNDS['lon_min'] <= lon <= BEIJING_BOUNDS['lon_max'])


def is_in_beijing_array(lat_array: np.ndarray, lon_array: np.ndarray) -> np.ndarray:
    """
    Verifica quais coordenadas estão dentro de Beijing.
    
    Args:
        lat_array: Array de latitudes
        lon_array: Array de longitudes
        
    Returns:
        np.ndarray: array de booleanos
    """
    in_lat = (lat_array >= BEIJING_BOUNDS['lat_min']) & (lat_array <= BEIJING_BOUNDS['lat_max'])
    in_lon = (lon_array >= BEIJING_BOUNDS['lon_min']) & (lon_array <= BEIJING_BOUNDS['lon_max'])
    return in_lat & in_lon