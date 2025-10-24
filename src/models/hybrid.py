"""
Hybrid Model: Ensemble of multiple models with learned weights
Implements Algorithms 12 and 13 from the paper
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class Swish(nn.Module):
    """Swish activation function"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class WeightingNetwork(nn.Module):
    """Neural network for learning model weights"""
    
    def __init__(self, feature_dim: int, num_models: int, hidden_dims: List[int]):
        super().__init__()
        
        layers = []
        input_dim = feature_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(Swish())
            input_dim = hidden_dim
        
        # Output layer (raw weights before softmax)
        layers.append(nn.Linear(input_dim, num_models))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: (batch, feature_dim)
        Returns:
            weights: (batch, num_models) - softmax weights
        """
        chi = self.net(x)  # Raw weights
        weights = torch.softmax(chi, dim=-1)  # Softmax normalization
        return weights


class HybridModel:
    """
    Hybrid Ensemble Model for bus arrival time estimation
    Combines multiple models with learned dynamic weights
    """
    
    def __init__(self, config: Dict, base_models: Dict):
        """
        Initialize Hybrid model
        
        Args:
            config: Configuration dictionary
            base_models: Dictionary of trained base models
                {'mst_av': model, 'gdrn_dft': model, ...}
        """
        self.n_m = config.get('n_m', 2)  # Top-n models to combine
        self.b = config.get('b', 6)  # Sinusoidal features
        self.hidden_dims = config.get('hidden_dims', [128, 64, 32, 16])
        self.num_epochs = config.get('num_epochs', 100)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 32)
        self.smoothing = config.get('smoothing', True)
        
        self.base_models = base_models
        self.model_names = list(base_models.keys())
        self.num_models = len(self.model_names)
        
        self.weighting_network = None
        self.feature_dim = None
        
    def train(self, training_data: List[Dict]):
        """
        Training Algorithm for Hybrid Model (Algorithm 12)
        
        Args:
            training_data: List of training examples, each with:
                {
                    'location': (lat, lon),
                    'weather': {...},
                    'time': datetime,
                    'speed': float,
                    'distance_to_stop': float,
                    'model_etas': {'mst_av': eta, 'gdrn_dft': eta, ...},
                    'actual_eta': float
                }
        """
        print("Training Hybrid model...")
        
        if len(training_data) == 0:
            print("Warning: No training data provided")
            return self
        
        # Step 1: Create feature vectors
        features = []
        model_etas_list = []
        actual_etas = []
        
        for example in training_data:
            feature_vec = self._create_feature_vector(
                example['location'],
                example['weather'],
                example['time'],
                example['speed'],
                example['distance_to_stop']
            )
            features.append(feature_vec)
            
            # Get ETAs from all models
            etas = [example['model_etas'].get(model_name, 0.0) 
                   for model_name in self.model_names]
            model_etas_list.append(etas)
            
            actual_etas.append(example['actual_eta'])
        
        features = np.array(features)
        model_etas_array = np.array(model_etas_list)
        actual_etas = np.array(actual_etas)
        
        self.feature_dim = features.shape[1]
        
        # Step 2: Initialize weighting network
        self.weighting_network = WeightingNetwork(
            self.feature_dim,
            self.num_models,
            self.hidden_dims
        )
        
        optimizer = torch.optim.Adam(
            self.weighting_network.parameters(),
            lr=self.learning_rate
        )
        criterion = nn.MSELoss()
        
        # Convert to tensors
        features_tensor = torch.FloatTensor(features)
        model_etas_tensor = torch.FloatTensor(model_etas_array)
        actual_etas_tensor = torch.FloatTensor(actual_etas)
        
        # Training loop
        dataset = torch.utils.data.TensorDataset(
            features_tensor, model_etas_tensor, actual_etas_tensor
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        
        for epoch in range(self.num_epochs):
            self.weighting_network.train()
            total_loss = 0.0
            
            for batch_features, batch_model_etas, batch_actual in dataloader:
                # Predict weights
                weights = self.weighting_network(batch_features)  # (batch, num_models)
                
                # Compute weighted ETA
                hybrid_eta = torch.sum(weights * batch_model_etas, dim=1)  # (batch,)
                
                # Compute loss
                loss = criterion(hybrid_eta, batch_actual)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}")
        
        print("Hybrid model training complete")
        return self
    
    def predict(self, location: Tuple[float, float],
                stop_location: Tuple[float, float],
                weather: Dict,
                time: datetime,
                speed: float,
                distance_to_stop: float,
                **kwargs) -> float:
        """
        Computation Algorithm for Hybrid Model (Algorithm 13)
        
        Args:
            location: Current bus location (lat, lon)
            stop_location: Target stop location (lat, lon)
            weather: Weather data dictionary
            time: Current datetime
            speed: Current speed
            distance_to_stop: Distance to stop in km
            **kwargs: Additional arguments for base models
        
        Returns:
            Hybrid ETA in minutes
        """
        if self.weighting_network is None:
            print("Warning: Hybrid model not trained")
            return 0.0
        
        # Step 1: Get predictions from all base models
        model_etas = {}
        
        for model_name, model in self.base_models.items():
            try:
                if hasattr(model, 'predict'):
                    eta = model.predict(location, stop_location, **kwargs)
                    model_etas[model_name] = eta
                else:
                    model_etas[model_name] = 0.0
            except Exception as e:
                print(f"Warning: Error predicting with {model_name}: {e}")
                model_etas[model_name] = 0.0
        
        # Step 2: Create feature vector
        feature_vec = self._create_feature_vector(
            location, weather, time, speed, distance_to_stop
        )
        
        # Step 3: Predict weights
        self.weighting_network.eval()
        with torch.no_grad():
            feature_tensor = torch.FloatTensor(feature_vec).unsqueeze(0)
            weights = self.weighting_network(feature_tensor).squeeze().numpy()
        
        # Step 4: Select top-n_m models
        top_indices = np.argsort(weights)[-self.n_m:]
        
        # Create mask for top models
        mask = np.zeros(self.num_models)
        mask[top_indices] = 1
        
        # Normalize weights for top models
        weights_masked = weights * mask
        weights_normalized = weights_masked / (np.sum(weights_masked) + 1e-8)
        
        # Step 5: Compute hybrid ETA
        hybrid_eta = 0.0
        for i, model_name in enumerate(self.model_names):
            hybrid_eta += weights_normalized[i] * model_etas[model_name]
        
        # Optional smoothing
        if self.smoothing and hybrid_eta > 0:
            # Simple exponential smoothing
            alpha = 0.3
            # Could maintain history for smoothing, simplified here
            hybrid_eta = hybrid_eta
        
        return hybrid_eta
    
    def _create_feature_vector(self, location: Tuple[float, float],
                               weather: Dict,
                               time: datetime,
                               speed: float,
                               distance_to_stop: float) -> np.ndarray:
        """
        Create feature vector for weighting network
        
        Features:
        - Location (lat, lon)
        - Weather (7 features)
        - Temporal sinusoidal features (b features)
        - Speed
        - Distance to stop
        """
        lat, lon = location
        
        # Weather features
        weather_features = [
            weather.get('humidity', 60),
            weather.get('precipitation', 0),
            weather.get('temperature', 298),
            weather.get('cloud_cover', 50),
            weather.get('pressure', 1013),
            weather.get('feels_like', 298),
            weather.get('wind_speed', 2.0)
        ]
        
        # Temporal features (sinusoidal encoding)
        hour = time.hour + time.minute / 60.0
        alpha = 2 * np.pi * hour / 24.0
        sin_features = [np.sin(alpha + 2 * m * np.pi / self.b) 
                       for m in range(self.b)]
        
        # Combine all features
        feature_vec = [lat, lon] + weather_features + sin_features + [speed, distance_to_stop]
        
        return np.array(feature_vec)
    
    def get_model_weights(self, location: Tuple[float, float],
                         weather: Dict,
                         time: datetime,
                         speed: float,
                         distance_to_stop: float) -> Dict[str, float]:
        """
        Get model weights for given context
        
        Returns:
            Dictionary mapping model names to weights
        """
        if self.weighting_network is None:
            return {name: 1.0 / self.num_models for name in self.model_names}
        
        feature_vec = self._create_feature_vector(
            location, weather, time, speed, distance_to_stop
        )
        
        self.weighting_network.eval()
        with torch.no_grad():
            feature_tensor = torch.FloatTensor(feature_vec).unsqueeze(0)
            weights = self.weighting_network(feature_tensor).squeeze().numpy()
        
        return {name: float(weights[i]) for i, name in enumerate(self.model_names)}
    
    def get_model_params(self) -> Dict:
        """Get model parameters for saving"""
        return {
            'weighting_network_state': self.weighting_network.state_dict() 
                                      if self.weighting_network else None,
            'model_names': self.model_names,
            'feature_dim': self.feature_dim,
            'config': {
                'n_m': self.n_m,
                'b': self.b,
                'hidden_dims': self.hidden_dims,
                'smoothing': self.smoothing
            }
        }
