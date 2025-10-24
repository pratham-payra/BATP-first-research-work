"""
GDRN-DFT Model: Graph Diffusion RNN with Discrete Fourier Transform
Implements Algorithms 4 and 5 from the paper
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from scipy.linalg import expm
from ..utils.haversine import haversine_distance


class GraphDiffusionRNN(nn.Module):
    """Bidirectional LSTM with graph diffusion for imputation"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_nodes: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim * 2, num_nodes)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, num_nodes)
        Returns:
            predictions: (batch, seq_len, num_nodes)
        """
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim*2)
        predictions = self.fc(lstm_out)  # (batch, seq_len, num_nodes)
        return predictions


class GDRNDFT:
    """
    GDRN-DFT Model for bus arrival time estimation
    Uses graph diffusion RNN for imputation and DFT for temporal patterns
    """
    
    def __init__(self, config: Dict):
        """
        Initialize GDRN-DFT model
        
        Args:
            config: Configuration dictionary with hyperparameters
        """
        self.delta_g = config.get('delta_g', 7)  # days
        self.delta_t = config.get('delta_t', 15)  # minutes
        self.tau_f = config.get('tau_f', 2)
        self.tau_b = config.get('tau_b', 2)
        self.d_g = config.get('d_g', 16)
        self.epsilon_g = config.get('epsilon_g', 0.1)
        self.num_epochs = config.get('num_epochs', 100)
        self.learning_rate = config.get('learning_rate', 0.001)
        
        self.model = None
        self.dft_amplitudes = {}  # node -> amplitudes
        self.dft_phases = {}  # node -> phases
        self.T = None  # Total time steps
        self.T_0 = None  # Reference time
        self.V_max = 80.0  # km/h
        self.epsilon = 0.1
        self.adjacency = None
        self.laplacian = None
        self.num_nodes = None
        
    def train(self, gps_data, adjacency_matrix, node_list):
        """
        Training Algorithm for GDRN-DFT (Algorithm 4)
        
        Args:
            gps_data: DataFrame with timestamps, grid_id, and speed
            adjacency_matrix: Adjacency matrix (n x n)
            node_list: List of node IDs
        """
        print("Training GDRN-DFT model...")
        
        self.adjacency = adjacency_matrix
        self.num_nodes = len(node_list)
        
        # Step 1: Compute normalized Laplacian
        self.laplacian = self._compute_laplacian(adjacency_matrix)
        
        # Step 2: Prepare time series data
        speed_matrix, mask_matrix, timestamps = self._prepare_time_series(
            gps_data, node_list
        )
        
        self.T = speed_matrix.shape[1]
        self.T_0 = timestamps[0]
        
        # Step 3: Initialize and train GDRN
        self.model = GraphDiffusionRNN(
            input_dim=self.num_nodes,
            hidden_dim=self.d_g,
            num_nodes=self.num_nodes
        )
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Convert to tensors
        speed_tensor = torch.FloatTensor(speed_matrix.T).unsqueeze(0)  # (1, T, n)
        mask_tensor = torch.FloatTensor(mask_matrix.T).unsqueeze(0)
        
        # Training loop
        for epoch in range(self.num_epochs):
            self.model.train()
            
            # Apply graph diffusion
            diffused = self._apply_diffusion(speed_tensor, mask_tensor)
            
            # Forward pass
            predictions = self.model(diffused)
            
            # Compute loss only on observed values
            loss = criterion(predictions * mask_tensor, speed_tensor * mask_tensor)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {loss.item():.4f}")
        
        # Step 4: Impute full dataset
        self.model.eval()
        with torch.no_grad():
            diffused = self._apply_diffusion(speed_tensor, mask_tensor)
            imputed = self.model(diffused)
            
            # Combine observed and imputed
            final_speeds = speed_tensor * mask_tensor + imputed * (1 - mask_tensor)
            final_speeds = final_speeds.squeeze(0).numpy().T  # (n, T)
        
        # Step 5: Compute DFT for each node
        for i, node_id in enumerate(node_list):
            time_series = final_speeds[i, :]
            
            # Compute DFT
            dft_coeffs = np.fft.fft(time_series)
            
            # Compute amplitudes and phases
            amplitudes = np.abs(dft_coeffs) / self.T
            phases = np.angle(dft_coeffs)
            
            # Apply cutoff threshold
            amplitudes[amplitudes < self.epsilon_g] = 0
            
            self.dft_amplitudes[node_id] = amplitudes
            self.dft_phases[node_id] = phases
        
        print(f"GDRN-DFT training complete. Computed DFT for {len(node_list)} nodes")
        
        return self
    
    def predict(self, bus_location: Tuple[float, float], 
                stop_location: Tuple[float, float],
                current_time: float, graph, node_attrs) -> float:
        """
        Computation Algorithm for GDRN-DFT (Algorithm 5)
        
        Args:
            bus_location: (latitude, longitude)
            stop_location: (latitude, longitude)
            current_time: Current timestamp
            graph: NetworkX graph
            node_attrs: Node attributes
        
        Returns:
            ETA in minutes
        """
        # Find nearest nodes
        bus_node = self._find_nearest_node(bus_location, node_attrs)
        stop_node = self._find_nearest_node(stop_location, node_attrs)
        
        if bus_node is None or stop_node is None:
            return 0.0
        
        # Compute shortest path
        try:
            path = nx.shortest_path(graph, bus_node, stop_node, weight='weight')
        except:
            return 0.0
        
        # Compute ETA
        eta = 0.0
        t = current_time
        
        for i in range(len(path) - 1):
            node_i = path[i]
            node_j = path[i + 1]
            
            # Compute time step
            t_s = (t - self.T_0) / (self.delta_t * 60)  # Convert to time steps
            
            # Reconstruct speeds using DFT
            speed_i = self._reconstruct_speed(node_i, t_s)
            speed_j = self._reconstruct_speed(node_j, t_s)
            
            # Filter speeds
            speed_i = np.clip(speed_i, self.epsilon, self.V_max)
            speed_j = np.clip(speed_j, self.epsilon, self.V_max)
            
            avg_speed = (speed_i + speed_j) / 2.0
            
            # Compute edge length
            lat_i = node_attrs[node_i]['median_lat']
            lon_i = node_attrs[node_i]['median_lon']
            lat_j = node_attrs[node_j]['median_lat']
            lon_j = node_attrs[node_j]['median_lon']
            
            edge_length = haversine_distance(lat_i, lon_i, lat_j, lon_j)
            
            # Update ETA and time
            travel_time = (edge_length / avg_speed) * 60.0  # minutes
            eta += travel_time
            t += travel_time * 60  # Convert back to seconds
        
        return eta
    
    def _compute_laplacian(self, adjacency: np.ndarray) -> np.ndarray:
        """Compute normalized Laplacian"""
        n = adjacency.shape[0]
        
        # Compute degree matrix
        degree = np.sum(adjacency, axis=1)
        degree[degree == 0] = 1  # Avoid division by zero
        
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
        
        # Normalized Laplacian: I - D^(-1/2) A D^(-1/2)
        L = np.eye(n) - D_inv_sqrt @ adjacency @ D_inv_sqrt
        
        return L
    
    def _apply_diffusion(self, speeds: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply graph diffusion for imputation"""
        # Compute diffusion operators
        L_np = self.laplacian
        
        # Forward diffusion: exp(-tau_f * L)
        diff_f = torch.FloatTensor(expm(-self.tau_f * L_np))
        
        # Backward diffusion: exp(-tau_b * L)
        diff_b = torch.FloatTensor(expm(-self.tau_b * L_np))
        
        # Average diffusion
        diff_avg = (diff_f + diff_b) / 2.0
        
        # Apply diffusion to each time step
        batch_size, T, n = speeds.shape
        diffused = torch.zeros_like(speeds)
        
        for t in range(T):
            diffused[:, t, :] = speeds[:, t, :] @ diff_avg.T
        
        # Mask and impute
        result = mask * speeds + (1 - mask) * diffused
        
        return result
    
    def _prepare_time_series(self, gps_data, node_list):
        """Prepare time series matrix from GPS data"""
        # Determine time range
        gps_data = gps_data.sort_values('timestamp')
        start_time = gps_data['timestamp'].min()
        end_time = gps_data['timestamp'].max()
        
        # Create time bins
        time_delta = pd.Timedelta(minutes=self.delta_t)
        timestamps = pd.date_range(start=start_time, end=end_time, freq=time_delta)
        
        T = len(timestamps)
        n = len(node_list)
        
        speed_matrix = np.zeros((n, T))
        mask_matrix = np.zeros((n, T))
        
        # Fill matrix
        for i, node_id in enumerate(node_list):
            node_data = gps_data[gps_data['grid_id'] == node_id]
            
            for t_idx, timestamp in enumerate(timestamps):
                # Find speeds in this time bin
                mask = (node_data['timestamp'] >= timestamp) & \
                       (node_data['timestamp'] < timestamp + time_delta)
                
                speeds_in_bin = node_data[mask]['speed'].values
                
                if len(speeds_in_bin) > 0:
                    speed_matrix[i, t_idx] = np.median(speeds_in_bin)
                    mask_matrix[i, t_idx] = 1
        
        return speed_matrix, mask_matrix, timestamps
    
    def _reconstruct_speed(self, node_id: str, t_s: float) -> float:
        """Reconstruct speed at time t_s using DFT"""
        if node_id not in self.dft_amplitudes:
            return 20.0  # Default speed
        
        amplitudes = self.dft_amplitudes[node_id]
        phases = self.dft_phases[node_id]
        
        # Reconstruct using inverse DFT formula
        speed = 0.0
        for k in range(len(amplitudes)):
            if amplitudes[k] > 0:
                speed += amplitudes[k] * np.cos(2 * np.pi * k * t_s / self.T + phases[k])
        
        return speed
    
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
            'dft_amplitudes': self.dft_amplitudes,
            'dft_phases': self.dft_phases,
            'T': self.T,
            'T_0': self.T_0,
            'config': {
                'delta_g': self.delta_g,
                'delta_t': self.delta_t,
                'tau_f': self.tau_f,
                'tau_b': self.tau_b,
                'd_g': self.d_g,
                'epsilon_g': self.epsilon_g
            }
        }


import pandas as pd
import networkx as nx
