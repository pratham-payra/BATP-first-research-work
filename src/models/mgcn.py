"""
MGCN Model: Masked Graph Convolutional Network
Implements Algorithms 10 and 11 from the paper
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from ..utils.haversine import haversine_distance


class MaskedGraphConvLayer(nn.Module):
    """Single layer of Masked Graph Convolution"""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.W_n = nn.Linear(in_features, out_features)  # Neighbor aggregation
        self.W_s = nn.Linear(in_features, out_features)  # Self features
        
    def forward(self, x, adj_norm, mask):
        """
        Args:
            x: (batch, num_nodes, in_features)
            adj_norm: Normalized adjacency matrix (num_nodes, num_nodes)
            mask: (batch, num_nodes, 1) - observation mask
        Returns:
            h: (batch, num_nodes, out_features)
        """
        # Masked neighbor aggregation
        masked_x = mask * x
        neighbor_agg = torch.matmul(adj_norm, masked_x)
        neighbor_features = self.W_n(neighbor_agg)
        
        # Self features
        self_features = self.W_s(x)
        
        # Combine and activate
        h = torch.relu(neighbor_features + self_features)
        
        return h


class MaskedGCN(nn.Module):
    """Multi-layer Masked Graph Convolutional Network"""
    
    def __init__(self, num_nodes: int, hidden_dims: list):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.layers = nn.ModuleList()
        
        # Build GCN layers
        input_dim = 1  # Single feature (speed)
        for hidden_dim in hidden_dims:
            self.layers.append(MaskedGraphConvLayer(input_dim, hidden_dim))
            input_dim = hidden_dim
        
        # Output layer
        self.W_out = nn.Linear(hidden_dims[-1], 1)
        
    def forward(self, x, adj_norm, mask):
        """
        Args:
            x: (batch, num_nodes, 1)
            adj_norm: (num_nodes, num_nodes)
            mask: (batch, num_nodes, 1)
        Returns:
            predictions: (batch, num_nodes, 1)
        """
        h = x
        
        # Apply GCN layers
        for layer in self.layers:
            h = layer(h, adj_norm, mask)
        
        # Output
        predictions = self.W_out(h)
        
        return predictions


class MGCN:
    """
    Masked Graph Convolutional Network for bus arrival time estimation
    Uses graph structure with masked observations and DFT priors
    """
    
    def __init__(self, config: Dict):
        """
        Initialize MGCN model
        
        Args:
            config: Configuration dictionary
        """
        self.delta_Tm = config.get('delta_Tm', 7)  # days
        self.delta_m = config.get('delta_m', 10)  # minutes
        self.L_m = config.get('L_m', 3)
        self.hidden_dims = config.get('hidden_dims', [64, 32, 16])
        self.lambda_m = config.get('lambda_m', 0.1)
        self.num_epochs = config.get('num_epochs', 150)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 16)
        
        self.model = None
        self.adj_norm = None
        self.num_nodes = None
        self.node_list = None
        
    def train(self, gps_data, adjacency_matrix, node_list, dft_speeds=None):
        """
        Training Algorithm for MGCN (Algorithm 10)
        
        Args:
            gps_data: DataFrame with timestamps, grid_id, and speed
            adjacency_matrix: Adjacency matrix (n x n)
            node_list: List of node IDs
            dft_speeds: Optional DFT-imputed speeds (n x T)
        """
        print("Training MGCN model...")
        
        self.num_nodes = len(node_list)
        self.node_list = node_list
        
        # Step 1: Compute normalized adjacency
        self.adj_norm = self._compute_normalized_adjacency(adjacency_matrix)
        
        # Step 2: Prepare time series data
        speed_matrix, mask_matrix, dft_matrix = self._prepare_time_series(
            gps_data, node_list, dft_speeds
        )
        
        T = speed_matrix.shape[1]
        
        # Step 3: Initialize model
        self.model = MaskedGCN(self.num_nodes, self.hidden_dims)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Convert to tensors
        adj_norm_tensor = torch.FloatTensor(self.adj_norm)
        
        # Training loop
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0
            
            # Train on sequences
            for t in range(T - 1):
                # Current state
                speed_t = torch.FloatTensor(speed_matrix[:, t]).unsqueeze(0).unsqueeze(-1)  # (1, n, 1)
                mask_t = torch.FloatTensor(mask_matrix[:, t]).unsqueeze(0).unsqueeze(-1)
                dft_t = torch.FloatTensor(dft_matrix[:, t]).unsqueeze(0).unsqueeze(-1)
                
                # Target (next time step)
                speed_t1 = torch.FloatTensor(speed_matrix[:, t + 1]).unsqueeze(0).unsqueeze(-1)
                mask_t1 = torch.FloatTensor(mask_matrix[:, t + 1]).unsqueeze(0).unsqueeze(-1)
                dft_t1 = torch.FloatTensor(dft_matrix[:, t + 1]).unsqueeze(0).unsqueeze(-1)
                
                # Form hybrid input
                hybrid_t = mask_t * speed_t + (1 - mask_t) * dft_t
                
                # Forward pass
                predictions = self.model(hybrid_t, adj_norm_tensor, mask_t)
                
                # Loss 1: Prediction loss on observed values
                loss1 = torch.sum(mask_t1 * (predictions - speed_t1) ** 2) / (torch.sum(mask_t1) + 1e-8)
                
                # Loss 2: Regularization with DFT prior
                loss2 = torch.mean((predictions - dft_t1) ** 2)
                
                # Total loss
                loss = loss1 + self.lambda_m * loss2
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / (T - 1)
                print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}")
        
        print("MGCN training complete")
        return self
    
    def predict(self, bus_location: Tuple[float, float],
                stop_location: Tuple[float, float],
                current_speeds: np.ndarray,
                mask: np.ndarray,
                dft_speeds: np.ndarray,
                graph, node_attrs) -> float:
        """
        Computation Algorithm for MGCN (Algorithm 11)
        
        Args:
            bus_location: (latitude, longitude)
            stop_location: (latitude, longitude)
            current_speeds: Current observed speeds (n,)
            mask: Observation mask (n,)
            dft_speeds: DFT-predicted speeds (n,)
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
        
        # Form hybrid input
        hybrid_speeds = mask * current_speeds + (1 - mask) * dft_speeds
        
        # Predict next speeds
        self.model.eval()
        with torch.no_grad():
            speed_tensor = torch.FloatTensor(hybrid_speeds).unsqueeze(0).unsqueeze(-1)
            mask_tensor = torch.FloatTensor(mask).unsqueeze(0).unsqueeze(-1)
            adj_norm_tensor = torch.FloatTensor(self.adj_norm)
            
            predictions = self.model(speed_tensor, adj_norm_tensor, mask_tensor)
            predicted_speeds = predictions.squeeze().numpy()
        
        # Compute ETA along path
        eta = 0.0
        
        for i in range(len(path) - 1):
            node_i = path[i]
            node_j = path[i + 1]
            
            # Get node indices
            idx_i = self.node_list.index(node_i) if node_i in self.node_list else 0
            idx_j = self.node_list.index(node_j) if node_j in self.node_list else 0
            
            # Get predicted speeds
            if mask[idx_i] == 1 and mask[idx_j] == 1:
                # Use predicted speeds for observed nodes
                speed_i = predicted_speeds[idx_i]
                speed_j = predicted_speeds[idx_j]
            else:
                # Use predicted speeds for masked nodes
                speed_i = predicted_speeds[idx_i]
                speed_j = predicted_speeds[idx_j]
            
            avg_speed = (speed_i + speed_j) / 2.0
            avg_speed = np.clip(avg_speed, 0.1, 80.0)
            
            # Compute edge length
            lat_i = node_attrs[node_i]['median_lat']
            lon_i = node_attrs[node_i]['median_lon']
            lat_j = node_attrs[node_j]['median_lat']
            lon_j = node_attrs[node_j]['median_lon']
            
            edge_length = haversine_distance(lat_i, lon_i, lat_j, lon_j)
            
            # Update ETA
            travel_time = (edge_length / avg_speed) * 60.0
            eta += travel_time
        
        return eta
    
    def _compute_normalized_adjacency(self, adjacency: np.ndarray) -> np.ndarray:
        """Compute normalized adjacency matrix with self-loops"""
        n = adjacency.shape[0]
        
        # Add self-loops
        A_tilde = adjacency + np.eye(n)
        
        # Compute degree matrix
        degree = np.sum(A_tilde, axis=1)
        degree[degree == 0] = 1
        
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
        
        # Normalized adjacency
        A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt
        
        return A_norm
    
    def _prepare_time_series(self, gps_data, node_list, dft_speeds):
        """Prepare time series matrices"""
        import pandas as pd
        
        gps_data = gps_data.sort_values('timestamp')
        
        # Create time bins
        start_time = gps_data['timestamp'].min()
        end_time = gps_data['timestamp'].max()
        time_delta = pd.Timedelta(minutes=self.delta_m)
        timestamps = pd.date_range(start=start_time, end=end_time, freq=time_delta)
        
        T = len(timestamps)
        n = len(node_list)
        
        speed_matrix = np.zeros((n, T))
        mask_matrix = np.zeros((n, T))
        dft_matrix = np.zeros((n, T))
        
        # Fill observed speeds
        for i, node_id in enumerate(node_list):
            node_data = gps_data[gps_data['grid_id'] == node_id]
            
            for t_idx, timestamp in enumerate(timestamps):
                mask = (node_data['timestamp'] >= timestamp) & \
                       (node_data['timestamp'] < timestamp + time_delta)
                
                speeds_in_bin = node_data[mask]['speed'].values
                
                if len(speeds_in_bin) > 0:
                    speed_matrix[i, t_idx] = np.median(speeds_in_bin)
                    mask_matrix[i, t_idx] = 1
                
                # DFT speeds (if provided)
                if dft_speeds is not None and i < dft_speeds.shape[0] and t_idx < dft_speeds.shape[1]:
                    dft_matrix[i, t_idx] = dft_speeds[i, t_idx]
                else:
                    dft_matrix[i, t_idx] = 20.0  # Default
        
        return speed_matrix, mask_matrix, dft_matrix
    
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
            'adj_norm': self.adj_norm,
            'node_list': self.node_list,
            'config': {
                'delta_Tm': self.delta_Tm,
                'delta_m': self.delta_m,
                'L_m': self.L_m,
                'hidden_dims': self.hidden_dims,
                'lambda_m': self.lambda_m
            }
        }
