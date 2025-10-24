"""
Data Acquisition Module
Implements Algorithm 1: Data Acquisition and Preprocessing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import requests
import os
from dotenv import load_dotenv
from .s3_manager import get_s3_manager

load_dotenv()


class DataAcquisition:
    """Handles data acquisition from S3 and external APIs"""
    
    def __init__(self, route: str):
        """
        Initialize data acquisition
        
        Args:
            route: Route identifier (e.g., 'route1')
        """
        self.route = route
        self.s3_manager = get_s3_manager()
        self.weather_api_key = os.getenv('OPENWEATHERMAP_API_KEY')
        self.weather_base_url = os.getenv('OPENWEATHERMAP_BASE_URL', 
                                          'https://history.openweathermap.org/data/2.5/history/city')
    
    def fetch_gps_traces(self, data_type: str = 'train') -> pd.DataFrame:
        """
        Fetch GPS traces from S3
        
        Args:
            data_type: 'train' or 'test'
        
        Returns:
            DataFrame with columns: timestamp, latitude, longitude, session_id, route
        """
        print(f"Fetching GPS traces for {self.route} ({data_type})...")
        gps_data = self.s3_manager.load_gps_data(self.route, data_type)
        
        # Ensure required columns exist
        required_cols = ['timestamp', 'latitude', 'longitude', 'Route', 'session_id']
        for col in required_cols:
            if col not in gps_data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Convert timestamp to datetime
        gps_data['timestamp'] = pd.to_datetime(gps_data['timestamp'])
        
        # Add Unix timestamp
        gps_data['unix_timestamp'] = gps_data['timestamp'].astype(np.int64) // 10**9
        
        print(f"Loaded {len(gps_data)} GPS records")
        return gps_data
    
    def fetch_bus_stops(self) -> pd.DataFrame:
        """
        Fetch bus stop dataset from S3
        
        Returns:
            DataFrame with columns: Name, Latitude, Longitude
        """
        print(f"Fetching bus stops for {self.route}...")
        bus_stops = self.s3_manager.load_bus_stops(self.route)
        
        required_cols = ['Name', 'Latitude', 'Longitude']
        for col in required_cols:
            if col not in bus_stops.columns:
                raise ValueError(f"Missing required column: {col}")
        
        print(f"Loaded {len(bus_stops)} bus stops")
        return bus_stops
    
    def fetch_weather_data(self, lat: float, lon: float, timestamp: datetime) -> Dict:
        """
        Fetch weather data from OpenWeatherMap API
        
        Args:
            lat: Latitude
            lon: Longitude
            timestamp: Timestamp for weather data
        
        Returns:
            Dictionary with weather parameters:
            - humidity (ρ): %
            - precipitation (κ): mm
            - temperature (ϖ): Kelvin
            - cloud_cover (μ): %
            - pressure (β): hPa
            - feels_like (ϖ_a): Kelvin
            - wind_speed (ν): m/s
        """
        # Check cache first
        date_str = timestamp.strftime('%Y-%m-%d')
        cached_weather = self.s3_manager.load_cached_weather(self.route, date_str)
        
        if cached_weather is not None:
            # Find closest timestamp in cache
            cached_weather['timestamp'] = pd.to_datetime(cached_weather['timestamp'])
            time_diff = abs(cached_weather['timestamp'] - timestamp)
            closest_idx = time_diff.idxmin()
            
            if time_diff[closest_idx] < timedelta(hours=1):
                return cached_weather.iloc[closest_idx].to_dict()
        
        # Fetch from API
        try:
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.weather_api_key,
                'type': 'hour',
                'start': int(timestamp.timestamp()),
                'end': int((timestamp + timedelta(hours=1)).timestamp())
            }
            
            response = requests.get(self.weather_base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'list' in data and len(data['list']) > 0:
                weather = data['list'][0]
                
                weather_dict = {
                    'timestamp': timestamp,
                    'latitude': lat,
                    'longitude': lon,
                    'humidity': weather['main'].get('humidity', 0),  # ρ (%)
                    'precipitation': weather.get('rain', {}).get('1h', 0),  # κ (mm)
                    'temperature': weather['main'].get('temp', 273.15),  # ϖ (K)
                    'cloud_cover': weather.get('clouds', {}).get('all', 0),  # μ (%)
                    'pressure': weather['main'].get('pressure', 1013),  # β (hPa)
                    'feels_like': weather['main'].get('feels_like', 273.15),  # ϖ_a (K)
                    'wind_speed': weather.get('wind', {}).get('speed', 0)  # ν (m/s)
                }
                
                return weather_dict
            else:
                return self._get_default_weather(lat, lon, timestamp)
                
        except Exception as e:
            print(f"Error fetching weather data: {e}")
            return self._get_default_weather(lat, lon, timestamp)
    
    def _get_default_weather(self, lat: float, lon: float, timestamp: datetime) -> Dict:
        """Return default weather values when API fails"""
        return {
            'timestamp': timestamp,
            'latitude': lat,
            'longitude': lon,
            'humidity': 60,
            'precipitation': 0,
            'temperature': 298.15,  # ~25°C
            'cloud_cover': 50,
            'pressure': 1013,
            'feels_like': 298.15,
            'wind_speed': 2.0
        }
    
    def fetch_weather_for_gps_data(self, gps_data: pd.DataFrame, 
                                   sample_rate: int = 10) -> pd.DataFrame:
        """
        Fetch weather data for all GPS points (with sampling to reduce API calls)
        
        Args:
            gps_data: GPS DataFrame
            sample_rate: Fetch weather every N records to reduce API calls
        
        Returns:
            DataFrame with weather data
        """
        print(f"Fetching weather data for GPS traces (sampling every {sample_rate} records)...")
        
        weather_records = []
        sampled_indices = range(0, len(gps_data), sample_rate)
        
        for idx in sampled_indices:
            row = gps_data.iloc[idx]
            weather = self.fetch_weather_data(
                row['latitude'],
                row['longitude'],
                row['timestamp']
            )
            weather_records.append(weather)
        
        weather_df = pd.DataFrame(weather_records)
        
        # Cache the weather data
        date_str = gps_data['timestamp'].iloc[0].strftime('%Y-%m-%d')
        self.s3_manager.cache_weather_data(weather_df, self.route, date_str)
        
        print(f"Fetched weather data for {len(weather_df)} points")
        return weather_df
    
    def interpolate_weather_data(self, gps_data: pd.DataFrame, 
                                 weather_data: pd.DataFrame) -> pd.DataFrame:
        """
        Interpolate weather data for all GPS points
        
        Args:
            gps_data: GPS DataFrame
            weather_data: Weather DataFrame (sampled)
        
        Returns:
            GPS DataFrame with interpolated weather columns
        """
        print("Interpolating weather data...")
        
        # Merge and interpolate
        gps_data = gps_data.sort_values('timestamp')
        weather_data = weather_data.sort_values('timestamp')
        
        # Merge on timestamp (nearest)
        gps_data = pd.merge_asof(
            gps_data,
            weather_data,
            on='timestamp',
            direction='nearest',
            suffixes=('', '_weather')
        )
        
        # Forward fill any missing values
        weather_cols = ['humidity', 'precipitation', 'temperature', 'cloud_cover', 
                       'pressure', 'feels_like', 'wind_speed']
        gps_data[weather_cols] = gps_data[weather_cols].fillna(method='ffill').fillna(method='bfill')
        
        print("Weather data interpolation complete")
        return gps_data
    
    def validate_data(self, gps_data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate data constraints
        
        Args:
            gps_data: GPS DataFrame with weather data
        
        Returns:
            Validated DataFrame
        """
        print("Validating data constraints...")
        
        # Validate temperature > 0
        gps_data.loc[gps_data['temperature'] <= 0, 'temperature'] = 273.15
        
        # Validate humidity in [0, 100]
        gps_data['humidity'] = gps_data['humidity'].clip(0, 100)
        
        # Validate cloud cover in [0, 100]
        gps_data['cloud_cover'] = gps_data['cloud_cover'].clip(0, 100)
        
        # Validate pressure > 0
        gps_data.loc[gps_data['pressure'] <= 0, 'pressure'] = 1013
        
        # Validate wind speed >= 0
        gps_data['wind_speed'] = gps_data['wind_speed'].clip(lower=0)
        
        print("Data validation complete")
        return gps_data
    
    def acquire_full_dataset(self, data_type: str = 'train', 
                            fetch_weather: bool = True) -> pd.DataFrame:
        """
        Complete data acquisition pipeline
        
        Args:
            data_type: 'train' or 'test'
            fetch_weather: Whether to fetch weather data
        
        Returns:
            Complete dataset with GPS and weather data
        """
        # Fetch GPS traces
        gps_data = self.fetch_gps_traces(data_type)
        
        # Fetch weather data if requested
        if fetch_weather:
            weather_data = self.fetch_weather_for_gps_data(gps_data)
            gps_data = self.interpolate_weather_data(gps_data, weather_data)
        
        # Validate data
        gps_data = self.validate_data(gps_data)
        
        print(f"Data acquisition complete: {len(gps_data)} records")
        return gps_data
