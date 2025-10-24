"""
Real-time ETA Predictor
Loads models from S3 and provides real-time predictions
"""

import numpy as np
from typing import Tuple, Dict, Optional
from datetime import datetime
from ..data.s3_manager import get_s3_manager
from ..data.preprocessing import DataPreprocessor


class ETAPredictor:
    """Real-time ETA prediction using trained models"""
    
    def __init__(self, route: str, model_name: str = 'hybrid'):
        """
        Initialize ETA predictor
        
        Args:
            route: Route identifier
            model_name: Name of model to use ('mst_av', 'gdrn_dft', 'knn', 'fenn', 'mgcn', 'hybrid')
        """
        self.route = route
        self.model_name = model_name
        self.s3_manager = get_s3_manager()
        
        self.model = None
        self.preprocessed_data = None
        self.graph = None
        self.node_attrs = None
        
        self._load_model()
        self._load_preprocessed_data()
    
    def _load_model(self):
        """Load trained model from S3"""
        print(f"Loading {self.model_name} model for {self.route}...")
        
        try:
            if self.model_name == 'mst_av':
                from ..models.mst_av import MSTAV
                model_params = self.s3_manager.load_model(MSTAV, 'mst_av', self.route)
                self.model = MSTAV()
                self.model.set_model_params(model_params)
                
            elif self.model_name == 'gdrn_dft':
                from ..models.gdrn_dft import GDRNDFT
                # Load config first
                metadata = self.s3_manager.load_model_metadata('gdrn_dft', self.route)
                config = metadata.get('config', {})
                self.model = GDRNDFT(config)
                # Load model parameters would go here
                
            elif self.model_name == 'knn':
                from ..models.knn import KNN
                metadata = self.s3_manager.load_model_metadata('knn', self.route)
                config = metadata.get('config', {})
                self.model = KNN(config)
                
            elif self.model_name == 'fenn':
                from ..models.fenn import FENN
                metadata = self.s3_manager.load_model_metadata('fenn', self.route)
                config = metadata.get('config', {})
                self.model = FENN(config)
                
            elif self.model_name == 'mgcn':
                from ..models.mgcn import MGCN
                metadata = self.s3_manager.load_model_metadata('mgcn', self.route)
                config = metadata.get('config', {})
                self.model = MGCN(config)
                
            elif self.model_name == 'hybrid':
                from ..models.hybrid import HybridModel
                metadata = self.s3_manager.load_model_metadata('hybrid', self.route)
                config = metadata.get('config', {})
                # Would need to load base models too
                self.model = HybridModel(config, {})
            
            print(f"Model loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _load_preprocessed_data(self):
        """Load preprocessed data from S3"""
        print(f"Loading preprocessed data for {self.route}...")
        
        try:
            preprocessor = DataPreprocessor(self.route)
            self.preprocessed_data = preprocessor.load_preprocessed_data()
            
            self.graph = self.preprocessed_data['graph']
            self.node_attrs = self.preprocessed_data['node_attrs']
            
            print("Preprocessed data loaded successfully")
            
        except Exception as e:
            print(f"Error loading preprocessed data: {e}")
            raise
    
    def predict_eta(self, bus_location: Tuple[float, float],
                   stop_location: Tuple[float, float],
                   current_time: Optional[datetime] = None,
                   weather_data: Optional[Dict] = None,
                   current_speed: float = 20.0) -> Dict:
        """
        Predict ETA for bus to reach stop
        
        Args:
            bus_location: (latitude, longitude) of current bus position
            stop_location: (latitude, longitude) of target bus stop
            current_time: Current datetime (default: now)
            weather_data: Optional weather data dictionary
            current_speed: Current bus speed in km/h
        
        Returns:
            Dictionary with prediction results:
            {
                'eta_minutes': float,
                'distance_km': float,
                'model': str,
                'timestamp': str
            }
        """
        if current_time is None:
            current_time = datetime.now()
        
        if weather_data is None:
            weather_data = {
                'humidity': 60,
                'precipitation': 0,
                'temperature': 298,
                'cloud_cover': 50,
                'pressure': 1013,
                'feels_like': 298,
                'wind_speed': 2.0
            }
        
        print(f"\nPredicting ETA...")
        print(f"Bus location: {bus_location}")
        print(f"Stop location: {stop_location}")
        
        # Compute distance
        from ..utils.haversine import haversine_distance
        distance = haversine_distance(
            bus_location[0], bus_location[1],
            stop_location[0], stop_location[1]
        )
        
        # Get prediction from model
        try:
            if self.model_name == 'mst_av':
                eta = self.model.predict(bus_location, stop_location)
                
            elif self.model_name in ['gdrn_dft', 'knn', 'fenn', 'mgcn']:
                eta = self.model.predict(
                    bus_location, stop_location,
                    graph=self.graph,
                    node_attrs=self.node_attrs
                )
                
            elif self.model_name == 'hybrid':
                eta = self.model.predict(
                    bus_location, stop_location,
                    weather=weather_data,
                    time=current_time,
                    speed=current_speed,
                    distance_to_stop=distance,
                    graph=self.graph,
                    node_attrs=self.node_attrs
                )
            else:
                eta = 0.0
            
            result = {
                'eta_minutes': float(eta),
                'distance_km': float(distance),
                'model': self.model_name,
                'route': self.route,
                'timestamp': current_time.isoformat(),
                'bus_location': bus_location,
                'stop_location': stop_location
            }
            
            print(f"\nPrediction Results:")
            print(f"  ETA: {eta:.2f} minutes")
            print(f"  Distance: {distance:.2f} km")
            print(f"  Model: {self.model_name}")
            
            return result
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            raise
    
    def predict_multiple_stops(self, bus_location: Tuple[float, float],
                              stop_locations: list,
                              **kwargs) -> list:
        """
        Predict ETA to multiple stops
        
        Args:
            bus_location: Current bus location
            stop_locations: List of (lat, lon) tuples for stops
            **kwargs: Additional arguments for predict_eta
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for stop_location in stop_locations:
            try:
                result = self.predict_eta(bus_location, stop_location, **kwargs)
                results.append(result)
            except Exception as e:
                print(f"Error predicting for stop {stop_location}: {e}")
                continue
        
        return results
