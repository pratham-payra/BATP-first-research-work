"""
KNN Model: Koopman Neural Network
Implements Algorithms 6 and 7 from the paper
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from ..utils.haversine import haversine_distance


class Swish(nn.Module):
    """Swish activation function"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class KoopmanEncoder(nn.Module):
    """Encoder MLP for lifting to Koopman space"""
    def __init__(self, input_dim: int, p: int, q: int, D_k: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, p),
            Swish(),
            nn.Linear(p, q),
            Swish(),
            nn.Linear(q, D_k)
        )
    
    def forward(self, x):
        return self.net(x)


class KoopmanDecoder(nn.Module):
    """Decoder MLP for reconstructing from Koopman space"""
    def __init__(self, D_k: int, q_prime: int, p_prime: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D_k, q_prime),
            Swish(),
            nn.Linear(q_prime, p_prime),
            Swish(),
            nn.Linear(p_prime, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class KoopmanNeuralNetwork(nn.Module):
    """Complete Koopman Neural Network"""
    def __init__(self, k: int, p: int, q: int, D_k: int, p_prime: int, q_prime: int):
        super().__init__()
        self.k = k
        self.D_k = D_k
        
        self.encoder = KoopmanEncoder(k, p, q, D_k)
        self.decoder = KoopmanDecoder(D_k, q_prime, p_prime, k)
        
        # Koopman matrix
        self.K = nn.Parameter(torch.randn(D_k, D_k) * 0.01)
    
    def forward(self, x, steps=1):
        """
        Args:
            x: (batch, k) - input sequence
            steps: Number of steps to predict ahead
        Returns:
            reconstruction: (batch, k)
            predictions: list of (batch, k) for each step
        """
        # Lift to Koopman space
        h = self.encoder(x)  # (batch, D_k)
        
        # Reconstruct
        x_recon = self.decoder(h)
        
        # Predict future states
        predictions = []
        h_pred = h
        for _ in range(steps):
            h_pred = h_pred @ self.K.T  # Apply Koopman operator
            x_pred = self.decoder(h_pred)
            predictions.append(x_pred)
        
        return x_recon, predictions


class KNN:
    """
    Koopman Neural Network for bus arrival time estimation
    Uses Koopman operator theory for temporal dynamics
    """
    
    def __init__(self, config: Dict):
        """
        Initialize KNN model
        
        Args:
            config: Configuration dictionary
        """
        self.d_k = config.get('d_k', 5)
        self.D_k = config.get('D_k', 12)
        self.delta_k = config.get('delta_k', 5)  # minutes
        self.p = config.get('p', 64)
        self.q = config.get('q', 32)
        self.p_prime = config.get('p_prime', 32)
        self.q_prime = config.get('q_prime', 64)
        self.lambda_k = config.get('lambda_k', 0.5)
        self.V_max = config.get('V_max', 80.0)
        self.epsilon = config.get('epsilon', 0.1)
        self.num_epochs = config.get('num_epochs', 200)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 32)
        
        self.model = None
        
    def train(self, gps_data):
        """
        Training Algorithm for KNN (Algorithm 6)
        
        Args:
            gps_data: DataFrame with timestamps and speeds
        """
        print("Training KNN model...")
        
        # Step 1: Aggregate speeds into time series
        time_series = self._aggregate_speeds(gps_data)
        
        # Step 2: Create training sequences
        X_train, y_train = self._create_sequences(time_series)
        
        if len(X_train) == 0:
            print("Warning: No training sequences created")
            return self
        
        # Step 3: Initialize model
        self.model = KoopmanNeuralNetwork(
            k=self.d_k,
            p=self.p,
            q=self.q,
            D_k=self.D_k,
            p_prime=self.p_prime,
            q_prime=self.q_prime
        )
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_train)
        y_tensor = torch.FloatTensor(y_train)
        
        # Training loop
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0
            
            for batch_x, batch_y in dataloader:
                # Forward pass
                x_recon, predictions = self.model(batch_x, steps=self.d_k)
                
                # Reconstruction loss
                recon_loss = torch.mean((batch_x - x_recon) ** 2)
                
                # Prediction loss
                pred_loss = 0.0
                for i, pred in enumerate(predictions):
                    if i < batch_y.shape[1]:
                        pred_loss += torch.mean((batch_y[:, i] - pred) ** 2)
                pred_loss /= len(predictions)
                
                # Total loss
                loss = recon_loss + self.lambda_k * pred_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}")
        
        print("KNN training complete")
        return self
    
    def predict(self, bus_location: Tuple[float, float],
                stop_location: Tuple[float, float],
                recent_speeds: list, graph, node_attrs) -> float:
        """
        Computation Algorithm for KNN (Algorithm 7)
        
        Args:
            bus_location: (latitude, longitude)
            stop_location: (latitude, longitude)
            recent_speeds: List of recent k speeds
            graph: NetworkX graph
            node_attrs: Node attributes
        
        Returns:
            ETA in minutes
        """
        if self.model is None:
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
        
        # Initialize ETA
        eta = 0.0
        remaining_distance = total_distance
        
        # Prepare initial speeds
        if len(recent_speeds) < self.d_k:
            recent_speeds = [20.0] * self.d_k  # Default speeds
        else:
            recent_speeds = recent_speeds[-self.d_k:]
        
        self.model.eval()
        with torch.no_grad():
            while remaining_distance > self.epsilon:
                # Apply logit transform
                speeds_filtered = [np.clip(v, self.epsilon, self.V_max - self.epsilon) 
                                  for v in recent_speeds]
                omega = [np.log(v / (self.V_max - v)) for v in speeds_filtered]
                
                # Predict next k speeds
                x_input = torch.FloatTensor([omega])
                _, predictions = self.model(x_input, steps=self.d_k)
                
                # Inverse logit transform
                predicted_omega = predictions[0].numpy()[0]  # Get first prediction
                predicted_speeds = [self.V_max / (1 + np.exp(-w)) for w in predicted_omega]
                
                # Update remaining distance and ETA
                for speed in predicted_speeds:
                    if remaining_distance <= self.epsilon:
                        break
                    
                    # Distance covered in delta_k minutes
                    distance_covered = (speed * self.delta_k) / 60.0  # km
                    
                    remaining_distance -= distance_covered
                    eta += self.delta_k
                
                # Update recent speeds for next iteration
                recent_speeds = predicted_speeds
        
        return eta
    
    def _aggregate_speeds(self, gps_data):
        """Aggregate speeds into time series at intervals delta_k"""
        gps_data = gps_data.sort_values('timestamp')
        
        # Create time bins
        start_time = gps_data['timestamp'].min()
        end_time = gps_data['timestamp'].max()
        
        import pandas as pd
        time_delta = pd.Timedelta(minutes=self.delta_k)
        timestamps = pd.date_range(start=start_time, end=end_time, freq=time_delta)
        
        speeds = []
        for i in range(len(timestamps) - 1):
            mask = (gps_data['timestamp'] >= timestamps[i]) & \
                   (gps_data['timestamp'] < timestamps[i + 1])
            
            speeds_in_bin = gps_data[mask]['speed'].values
            
            if len(speeds_in_bin) > 0:
                speeds.append(np.median(speeds_in_bin))
            else:
                speeds.append(20.0)  # Default
        
        return np.array(speeds)
    
    def _create_sequences(self, time_series):
        """Create input-output sequences for training"""
        X = []
        y = []
        
        for i in range(len(time_series) - 2 * self.d_k):
            # Input: k past speeds
            input_speeds = time_series[i:i + self.d_k]
            
            # Filter and transform
            input_filtered = np.clip(input_speeds, self.epsilon, self.V_max - self.epsilon)
            input_omega = np.log(input_filtered / (self.V_max - input_filtered))
            
            # Target: next k speeds
            target_speeds = time_series[i + self.d_k:i + 2 * self.d_k]
            target_filtered = np.clip(target_speeds, self.epsilon, self.V_max - self.epsilon)
            target_omega = np.log(target_filtered / (self.V_max - target_filtered))
            
            X.append(input_omega)
            y.append(target_omega)
        
        return np.array(X), np.array(y)
    
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
            'model_state': self.model.state_dict() if self.model else None,
            'config': {
                'd_k': self.d_k,
                'D_k': self.D_k,
                'delta_k': self.delta_k,
                'p': self.p,
                'q': self.q,
                'p_prime': self.p_prime,
                'q_prime': self.q_prime,
                'lambda_k': self.lambda_k,
                'V_max': self.V_max,
                'epsilon': self.epsilon
            }
        }
