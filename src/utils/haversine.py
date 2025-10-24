"""
Haversine Distance Calculation
Implements the Haversine formula for computing distances between GPS coordinates
"""

import numpy as np
from typing import Union


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the Haversine distance between two points on Earth
    
    Formula from Algorithm 1:
    a = sin²((θ₂ - θ₁)/2) + cos(θ₁)cos(θ₂)sin²((φ₂ - φ₁)/2)
    c = 2·arctan2(√a, √(1-a))
    d = R·c  (where R = 6371 km)
    
    Args:
        lat1: Latitude of point 1 (degrees)
        lon1: Longitude of point 1 (degrees)
        lat2: Latitude of point 2 (degrees)
        lon2: Longitude of point 2 (degrees)
    
    Returns:
        Distance in kilometers
    """
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # Earth radius in kilometers
    R = 6371.0
    
    distance = R * c
    
    return distance


def haversine_distance_array(lat1: np.ndarray, lon1: np.ndarray, 
                             lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """
    Vectorized Haversine distance calculation for arrays
    
    Args:
        lat1: Array of latitudes for point 1
        lon1: Array of longitudes for point 1
        lat2: Array of latitudes for point 2
        lon2: Array of longitudes for point 2
    
    Returns:
        Array of distances in kilometers
    """
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # Earth radius in kilometers
    R = 6371.0
    
    distances = R * c
    
    return distances


def meters_to_degrees_lat(meters: float) -> float:
    """
    Convert meters to degrees latitude
    
    Args:
        meters: Distance in meters
    
    Returns:
        Distance in degrees latitude
    """
    # 1 degree latitude ≈ 111,000 meters
    return meters / 111000.0


def meters_to_degrees_lon(meters: float, latitude: float) -> float:
    """
    Convert meters to degrees longitude at a given latitude
    
    Args:
        meters: Distance in meters
        latitude: Latitude in degrees
    
    Returns:
        Distance in degrees longitude
    """
    # 1 degree longitude ≈ 111,000 * cos(latitude) meters
    return meters / (111000.0 * np.cos(np.radians(latitude)))
