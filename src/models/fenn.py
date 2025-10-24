"""
FE-NN Model: Feature-Encoded Neural Network
Implements Algorithms 8 and 9 from the paper
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
from datetime import datetime
from ..utils.haversine import haversine_distance


class Swish(nn.Module):
    """Swish activation function"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class FeatureEncoder(nn.Module):
    """Encoder for feature vectors"""
    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.fc(x)


class FeatureEncodedNN(nn.Module):
    """Neural network for speed prediction from encoded features"""
    def __init__(self, k: int, hidden_dims: List[int]):
        super().__init__()
        
        layers = []
        input_dim = k
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(Swish())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, k))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class FENN:
    """
    Feature-Encoded Neural Network for bus arrival time estimation
    Uses location, weather, and temporal features
    """
    
    def __init__(self, config: Dict):
        """
        Initialize FE-NN model
        
        Args:
            config: Configuration dictionary
        """
        self.k = config.get('k', 7)
        self.delta_n = config.get('delta_n', 10)  # minutes
        self.b = config.get('b', 6)
        self.l_f = config.get('l_f', 3)
        self.hidden_dims = config.get('hidden_dims', [128, 64, 32])
        self.V_max = config.get('V_max', 80.0)
        self.epsilon = config.get('epsilon', 0.1)
        self.num_epochs = config.get('num_epochs', 150)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 32)
        
        self.encoder = None
        self.model = None
        self.feature_dim = None
        
    def train(self, gps_data):
        """
        Training Algorithm for FE-NN (Algorithm 8)
        
        Args:
            gps_data: DataFrame with location, weather, time, and speed data
        """
        print("Training FE-NN model...")
        
        # Step 1: Create feature vectors
        features, targets = self._create_features(gps_data)
        
        if len(features) == 0:
            print("Warning: No training features created")
            return self
        
        self.feature_dim = features[0].shape[0]
        
        # Step 2: Initialize encoder and neural network
        self.encoder = FeatureEncoder(self.feature_dim, output_dim=1)
        self.model = FeatureEncodedNN(self.k, self.hidden_dims)
        
        # Combine parameters for optimization
        params = list(self.encoder.parameters()) + list(self.model.parameters())
        optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Convert to tensors
        X_features = torch.FloatTensor(np.array(features))  # (N, k, feature_dim)
        y_targets = torch.FloatTensor(np.array(targets))    # (N, k)
        
        # Training loop
        dataset = torch.utils.data.TensorDataset(X_features, y_targets)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        
        for epoch in range(self.num_epochs):
            self.encoder.train()
            self.model.train()
            total_loss = 0.0
            
            for batch_features, batch_targets in dataloader:
                batch_size, k, feat_dim = batch_features.shape
                
                # Encode each time step
                encoded = []
                for t in range(k):
                    enc_t = self.encoder(batch_features[:, t, :])  # (batch, 1)
                    encoded.append(enc_t)
                
                encoded_input = torch.cat(encoded, dim=1)  # (batch, k)
                
                # Predict
                predictions = self.model(encoded_input)
                
                # Compute loss
                loss = criterion(predictions, batch_targets)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}")
        
        print("FE-NN training complete")
        return self
    
    def predict(self, bus_location: Tuple[float, float],
                stop_location: Tuple[float, float],
                current_time: datetime,
                weather_data: Dict,
                recent_speeds: List[float],
                graph, node_attrs) -> float:
        """
        Computation Algorithm for FE-NN (Algorithm 9)
        
        Args:
            bus_location: (latitude, longitude)
            stop_location: (latitude, longitude)
            current_time: Current datetime
            weather_data: Dictionary with weather parameters
            recent_speeds: List of recent speeds
            graph: NetworkX graph
            node_attrs: Node attributes
        
        Returns:
            ETA in minutes
        """
        if self.model is None or self.encoder is None:
            return 0.0
        
        # Find path
        bus_node = self._find_nearest_node(bus_location, node_attrs)
        stop_node = self._find_nearest_node(stop_location, node_attrs)
        
        if bus_node is None or stop_node is None:
            return 0.0
        
        try:
            import networkx as nx
            path = nx.shortest_path(graph, bus_node, stop_node, weight='weight')
        except:
            return 0.0
        
        # Compute total distance
        total_distance = 0.0
        for i in range(len(path) - 1):
            node_i = path[i]
            node_j = path[i + 1]
            
            lat_i = node_attrs[node_i]['median_lat']
            lon_i = node_attrs[node_i]['median_lon']
            lat_j = node_attrs[node_j]['median_lat']
            lon_j = node_attrs[node_j]['median_lon']
            
            total_distance += haversine_distance(lat_i, lon_i, lat_j, lon_j)
        
        # Initialize
        eta = 0.0
        remaining_distance = total_distance
        current_location = bus_location
        
        # Prepare initial features
        if len(recent_speeds) < self.k:
            recent_speeds = [20.0] * self.k
        else:
            recent_speeds = recent_speeds[-self.k:]
        
        self.encoder.eval()
        self.model.eval()
        
        with torch.no_grad():
            while remaining_distance > self.epsilon:
                # Create feature matrix for current window
                features = []
                for i in range(self.k):
                    time_offset = i * self.delta_n
                    feature_vec = self._create_feature_vector(
                        current_location,
                        current_time,
                        weather_data,
                        recent_speeds[i] if i < len(recent_speeds) else 20.0
                    )
                    features.append(feature_vec)
                
                features_tensor = torch.FloatTensor(features).unsqueeze(0)  # (1, k, feat_dim)
                
                # Encode features
                encoded = []
                for t in range(self.k):
                    enc_t = self.encoder(features_tensor[:, t, :])
                    encoded.append(enc_t)
                
                encoded_input = torch.cat(encoded, dim=1)  # (1, k)
                
                # Predict speeds
                pred_omega = self.model(encoded_input).numpy()[0]
                
                # Inverse logit transform
                predicted_speeds = [self.V_max / (1 + np.exp(-w)) for w in pred_omega]
                
                # Update remaining distance and ETA
                for speed in predicted_speeds:
                    if remaining_distance <= self.epsilon:
                        break
                    
                    distance_covered = (speed * self.delta_n) / 60.0  # km
                    remaining_distance -= distance_covered
                    eta += self.delta_n
                
                # Update for next iteration
                recent_speeds = predicted_speeds
                # Update location (simplified - move along path)
                # In practice, you'd track position along the path
        
        return eta
    
    def _create_features(self, gps_data):
        """Create feature vectors and targets from GPS data"""
        import pandas as pd
        
        gps_data = gps_data.sort_values('timestamp')
        
        # Create time bins
        start_time = gps_data['timestamp'].min()
        end_time = gps_data['timestamp'].max()
        time_delta = pd.Timedelta(minutes=self.delta_n)
        timestamps = pd.date_range(start=start_time, end=end_time, freq=time_delta)
        
        features = []
        targets = []
        
        for i in range(len(timestamps) - 2 * self.k):
            # Create feature matrix for k time steps
            feature_matrix = []
            speed_sequence = []
            
            valid = True
            for j in range(self.k):
                t_idx = i + j
                if t_idx >= len(timestamps):
                    valid = False
                    break
                
                timestamp = timestamps[t_idx]
                
                # Get data in this time bin
                mask = (gps_data['timestamp'] >= timestamp) & \
                       (gps_data['timestamp'] < timestamp + time_delta)
                
                bin_data = gps_data[mask]
                
                if len(bin_data) == 0:
                    valid = False
                    break
                
                # Extract features
                lat = bin_data['latitude'].median()
                lon = bin_data['longitude'].median()
                speed = bin_data['speed'].median()
                
                # Weather features (if available)
                weather_features = []
                if 'humidity' in bin_data.columns:
                    weather_features = [
                        bin_data['humidity'].median(),
                        bin_data['precipitation'].median(),
                        bin_data['temperature'].median(),
                        bin_data['cloud_cover'].median(),
                        bin_data['pressure'].median(),
                        bin_data['feels_like'].median(),
                        bin_data['wind_speed'].median()
                    ]
                else:
                    weather_features = [60, 0, 298, 50, 1013, 298, 2.0]  # Defaults
                
                # Temporal features (sinusoidal)
                hour = timestamp.hour + timestamp.minute / 60.0
                alpha = 2 * np.pi * hour / 24.0
                sin_features = [np.sin(alpha + 2 * m * np.pi / self.b) 
                               for m in range(self.b)]
                
                # Combine features
                feature_vec = [lat, lon] + weather_features + sin_features + [speed]
                feature_matrix.append(feature_vec)
                speed_sequence.append(speed)
            
            if not valid:
                continue
            
            # Create target sequence (next k speeds)
            target_sequence = []
            for j in range(self.k):
                t_idx = i + self.k + j
                if t_idx >= len(timestamps):
                    valid = False
                    break
                
                timestamp = timestamps[t_idx]
                mask = (gps_data['timestamp'] >= timestamp) & \
                       (gps_data['timestamp'] < timestamp + time_delta)
                
                bin_data = gps_data[mask]
                
                if len(bin_data) == 0:
                    valid = False
                    break
                
                speed = bin_data['speed'].median()
                
                # Filter and logit transform
                speed_filtered = np.clip(speed, self.epsilon, self.V_max - self.epsilon)
                omega = np.log(speed_filtered / (self.V_max - speed_filtered))
                target_sequence.append(omega)
            
            if valid and len(target_sequence) == self.k:
                features.append(feature_matrix)
                targets.append(target_sequence)
        
        return features, targets
    
    def _create_feature_vector(self, location: Tuple[float, float],
                               timestamp: datetime,
                               weather: Dict,
                               speed: float) -> np.ndarray:
        """Create feature vector for a single time step"""
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
        
        # Temporal features
        hour = timestamp.hour + timestamp.minute / 60.0
        alpha = 2 * np.pi * hour / 24.0
        sin_features = [np.sin(alpha + 2 * m * np.pi / self.b) 
                       for m in range(self.b)]
        
        # Combine
        feature_vec = [lat, lon] + weather_features + sin_features + [speed]
        
        return np.array(feature_vec)
    
    def _find_nearest_node(self, location: Tuple[float, float],
                          node_attrs: Dict) -> Optional[str]:
        """Find nearest node to location"""
        lat, lon = location
        min_dist = float('inf')
        nearest = None
        
        for node_id, attrs in node_attrs.items():
            dist = haversine_distance(lat, lon, attrs['median_lat'], attrs['median_lon'])
            if dist < min_dist:
                min_dist = dist
                nearest = node_id
        
        return nearest
    
    def get_model_params(self) -> Dict:
        """Get model parameters for saving"""
        return {
            'encoder_state': self.encoder.state_dict() if self.encoder else None,
            'model_state': self.model.state_dict() if self.model else None,
            'feature_dim': self.feature_dim,
            'config': {
                'k': self.k,
                'delta_n': self.delta_n,
                'b': self.b,
                'l_f': self.l_f,
                'hidden_dims': self.hidden_dims,
                'V_max': self.V_max,
                'epsilon': self.epsilon
            }
        }
