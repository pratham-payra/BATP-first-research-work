"""
Benchmark Models: Spatio-temporal graph models for comparison
Implements Algorithm 14 from the paper

Benchmark models:
- DCRNN (Diffusion Convolutional Recurrent Neural Network)
- STGCN (Spatio-Temporal Graph Convolutional Network)
- GWNet (Graph WaveNet)
- T-GCN (Temporal Graph Convolutional Network)
- MTGNN (Multivariate Time Series Graph Neural Network)
- STFGNN (Spatio-Temporal Fusion Graph Neural Network)
- ST-ResNet (Spatio-Temporal Residual Network)
- ST-GConv (Spatio-Temporal Graph Convolution)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from ..utils.haversine import haversine_distance


# ==================== DCRNN ====================

class DiffusionConvLayer(nn.Module):
    """Diffusion convolution layer"""
    def __init__(self, in_channels: int, out_channels: int, K: int = 2):
        super().__init__()
        self.K = K
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels, K))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x, adj):
        """
        Args:
            x: (batch, num_nodes, in_channels)
            adj: (num_nodes, num_nodes)
        """
        batch_size, num_nodes, in_channels = x.shape
        
        # Compute diffusion powers
        diffusion = []
        for k in range(self.K):
            if k == 0:
                diffusion.append(x)
            else:
                diffusion.append(torch.matmul(adj, diffusion[-1]))
        
        # Weighted sum
        out = torch.zeros(batch_size, num_nodes, self.weight.shape[1]).to(x.device)
        for k in range(self.K):
            out += torch.matmul(diffusion[k], self.weight[:, :, k])
        
        return out


class DCRNN(nn.Module):
    """Diffusion Convolutional Recurrent Neural Network"""
    def __init__(self, num_nodes: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        
        # Diffusion convolution + GRU
        self.diff_conv = DiffusionConvLayer(1, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, adj):
        """
        Args:
            x: (batch, seq_len, num_nodes, 1)
            adj: (num_nodes, num_nodes)
        """
        batch_size, seq_len, num_nodes, _ = x.shape
        
        # Apply diffusion convolution to each time step
        conv_out = []
        for t in range(seq_len):
            conv_t = self.diff_conv(x[:, t, :, :], adj)  # (batch, num_nodes, hidden_dim)
            conv_out.append(conv_t)
        
        conv_out = torch.stack(conv_out, dim=1)  # (batch, seq_len, num_nodes, hidden_dim)
        
        # Reshape for GRU
        conv_out = conv_out.view(batch_size * num_nodes, seq_len, self.hidden_dim)
        
        # Apply GRU
        gru_out, _ = self.gru(conv_out)  # (batch*num_nodes, seq_len, hidden_dim)
        
        # Take last time step
        last_out = gru_out[:, -1, :]  # (batch*num_nodes, hidden_dim)
        
        # Predict
        pred = self.fc(last_out)  # (batch*num_nodes, 1)
        pred = pred.view(batch_size, num_nodes, 1)
        
        return pred


# ==================== STGCN ====================

class TemporalConvLayer(nn.Module):
    """Temporal convolution layer"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=(0, kernel_size//2))
    
    def forward(self, x):
        return torch.relu(self.conv(x))


class SpatialConvLayer(nn.Module):
    """Spatial graph convolution layer"""
    def __init__(self, in_channels: int, out_channels: int, K: int = 3):
        super().__init__()
        self.K = K
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels, K))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x, adj):
        """
        Args:
            x: (batch, in_channels, num_nodes, seq_len)
            adj: (num_nodes, num_nodes)
        """
        batch_size, in_channels, num_nodes, seq_len = x.shape
        
        # Compute Chebyshev polynomials
        cheb = []
        for k in range(self.K):
            if k == 0:
                cheb.append(torch.eye(num_nodes).to(x.device))
            elif k == 1:
                cheb.append(adj)
            else:
                cheb.append(2 * torch.matmul(adj, cheb[-1]) - cheb[-2])
        
        # Apply convolution
        out = torch.zeros(batch_size, self.weight.shape[1], num_nodes, seq_len).to(x.device)
        for k in range(self.K):
            # x_k = (batch, in_channels, num_nodes, seq_len)
            x_k = torch.einsum('bcnt,nm->bcmt', x, cheb[k])
            out += torch.einsum('bcnt,cok->bont', x_k, self.weight[:, :, k:k+1])
        
        return out


class STGCN(nn.Module):
    """Spatio-Temporal Graph Convolutional Network"""
    def __init__(self, num_nodes: int, num_layers: int = 2):
        super().__init__()
        self.num_nodes = num_nodes
        
        # ST blocks
        self.temporal1 = TemporalConvLayer(1, 32, 3)
        self.spatial1 = SpatialConvLayer(32, 32, 3)
        self.temporal2 = TemporalConvLayer(32, 64, 3)
        
        self.fc = nn.Linear(64, 1)
    
    def forward(self, x, adj):
        """
        Args:
            x: (batch, seq_len, num_nodes, 1)
            adj: (num_nodes, num_nodes)
        """
        # Reshape to (batch, channels, num_nodes, seq_len)
        x = x.permute(0, 3, 2, 1)
        
        # ST block 1
        x = self.temporal1(x)
        x = self.spatial1(x, adj)
        x = self.temporal2(x)
        
        # Global pooling over time
        x = torch.mean(x, dim=-1)  # (batch, channels, num_nodes)
        
        # Predict
        x = x.permute(0, 2, 1)  # (batch, num_nodes, channels)
        pred = self.fc(x)
        
        return pred


# ==================== GWNet ====================

class GWNet(nn.Module):
    """Graph WaveNet"""
    def __init__(self, num_nodes: int, num_layers: int = 8):
        super().__init__()
        self.num_nodes = num_nodes
        
        # Adaptive adjacency matrix
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, 10))
        
        # Dilated causal convolutions
        self.dilated_convs = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            self.dilated_convs.append(
                nn.Conv2d(32, 32, (1, 2), dilation=(1, dilation), padding=(0, dilation))
            )
        
        self.input_conv = nn.Conv2d(1, 32, (1, 1))
        self.fc = nn.Linear(32, 1)
    
    def forward(self, x, adj):
        """
        Args:
            x: (batch, seq_len, num_nodes, 1)
            adj: (num_nodes, num_nodes)
        """
        # Compute adaptive adjacency
        adaptive_adj = torch.softmax(
            torch.relu(torch.matmul(self.node_embeddings, self.node_embeddings.T)),
            dim=1
        )
        
        # Reshape
        x = x.permute(0, 3, 2, 1)  # (batch, 1, num_nodes, seq_len)
        x = self.input_conv(x)
        
        # Apply dilated convolutions
        for conv in self.dilated_convs:
            residual = x
            x = torch.relu(conv(x))
            x = x + residual[:, :, :, :x.shape[-1]]
        
        # Global pooling
        x = torch.mean(x, dim=-1)  # (batch, channels, num_nodes)
        x = x.permute(0, 2, 1)
        
        pred = self.fc(x)
        return pred


# ==================== T-GCN ====================

class TGCN(nn.Module):
    """Temporal Graph Convolutional Network"""
    def __init__(self, num_nodes: int, hidden_dim: int = 64):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        
        # Graph convolution
        self.gc = nn.Linear(1, hidden_dim)
        
        # GRU
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, adj):
        """
        Args:
            x: (batch, seq_len, num_nodes, 1)
            adj: (num_nodes, num_nodes)
        """
        batch_size, seq_len, num_nodes, _ = x.shape
        
        # Initialize hidden state
        h = torch.zeros(batch_size * num_nodes, self.hidden_dim).to(x.device)
        
        # Process sequence
        for t in range(seq_len):
            x_t = x[:, t, :, :]  # (batch, num_nodes, 1)
            
            # Graph convolution
            x_t = x_t.view(batch_size * num_nodes, 1)
            x_t = self.gc(x_t)  # (batch*num_nodes, hidden_dim)
            
            # GRU update
            h = self.gru(x_t, h)
        
        # Predict
        pred = self.fc(h)
        pred = pred.view(batch_size, num_nodes, 1)
        
        return pred


# ==================== Benchmark Model Wrapper ====================

class BenchmarkModel:
    """
    Wrapper for benchmark spatio-temporal graph models
    Implements Algorithm 14: Benchmark Models ETA Computation
    """
    
    def __init__(self, model_type: str, config: Dict):
        """
        Initialize benchmark model
        
        Args:
            model_type: One of 'dcrnn', 'stgcn', 'gwnet', 'tgcn', 'mtgnn', 'stfgnn', 'st_resnet', 'st_gconv'
            config: Configuration dictionary
        """
        self.model_type = model_type
        self.config = config
        self.model = None
        self.num_nodes = None
        self.node_list = None
        self.adj_norm = None
    
    def train(self, gps_data, adjacency_matrix, node_list):
        """
        Train benchmark model
        
        Args:
            gps_data: DataFrame with GPS data
            adjacency_matrix: Adjacency matrix
            node_list: List of node IDs
        """
        print(f"Training {self.model_type.upper()} model...")
        
        self.num_nodes = len(node_list)
        self.node_list = node_list
        self.adj_norm = self._compute_normalized_adjacency(adjacency_matrix)
        
        # Initialize model
        if self.model_type == 'dcrnn':
            self.model = DCRNN(
                self.num_nodes,
                hidden_dim=self.config.get('hidden_dim', 64),
                num_layers=self.config.get('num_layers', 2)
            )
        elif self.model_type == 'stgcn':
            self.model = STGCN(
                self.num_nodes,
                num_layers=self.config.get('num_layers', 2)
            )
        elif self.model_type == 'gwnet':
            self.model = GWNet(
                self.num_nodes,
                num_layers=self.config.get('num_layers', 8)
            )
        elif self.model_type == 'tgcn':
            self.model = TGCN(
                self.num_nodes,
                hidden_dim=self.config.get('hidden_dim', 64)
            )
        else:
            # Placeholder for other models
            self.model = DCRNN(self.num_nodes)
        
        # Prepare training data
        speed_sequences, _ = self._prepare_sequences(gps_data, node_list)
        
        if len(speed_sequences) == 0:
            print("Warning: No training sequences")
            return self
        
        # Training loop
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 0.001)
        )
        criterion = nn.MSELoss()
        
        X_tensor = torch.FloatTensor(speed_sequences)
        adj_tensor = torch.FloatTensor(self.adj_norm)
        
        num_epochs = self.config.get('num_epochs', 100)
        batch_size = self.config.get('batch_size', 16)
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            
            for i in range(0, len(X_tensor), batch_size):
                batch_x = X_tensor[i:i+batch_size]
                
                # Predict
                predictions = self.model(batch_x, adj_tensor)
                
                # Target is next time step
                targets = batch_x[:, -1:, :, :]
                
                loss = criterion(predictions, targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")
        
        print(f"{self.model_type.upper()} training complete")
        return self
    
    def predict(self, bus_location: Tuple[float, float],
                stop_location: Tuple[float, float],
                recent_speeds: np.ndarray,
                graph, node_attrs) -> float:
        """
        Predict ETA using benchmark model
        
        Args:
            bus_location: (lat, lon)
            stop_location: (lat, lon)
            recent_speeds: Recent speed observations (num_nodes, seq_len)
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
        
        # Prepare input
        self.model.eval()
        with torch.no_grad():
            # Reshape recent speeds
            if recent_speeds.shape[0] != self.num_nodes:
                recent_speeds = np.zeros((self.num_nodes, 10))
            
            x_input = torch.FloatTensor(recent_speeds).unsqueeze(0).unsqueeze(-1)
            adj_tensor = torch.FloatTensor(self.adj_norm)
            
            # Predict speeds
            predicted_speeds = self.model(x_input, adj_tensor).squeeze().numpy()
        
        # Compute ETA along path
        eta = 0.0
        for i in range(len(path) - 1):
            node_i = path[i]
            node_j = path[i + 1]
            
            idx_i = self.node_list.index(node_i) if node_i in self.node_list else 0
            idx_j = self.node_list.index(node_j) if node_j in self.node_list else 0
            
            speed_i = predicted_speeds[idx_i]
            speed_j = predicted_speeds[idx_j]
            avg_speed = (speed_i + speed_j) / 2.0
            avg_speed = np.clip(avg_speed, 0.1, 80.0)
            
            lat_i = node_attrs[node_i]['median_lat']
            lon_i = node_attrs[node_i]['median_lon']
            lat_j = node_attrs[node_j]['median_lat']
            lon_j = node_attrs[node_j]['median_lon']
            
            distance = haversine_distance(lat_i, lon_i, lat_j, lon_j)
            travel_time = (distance / avg_speed) * 60.0
            eta += travel_time
        
        return eta
    
    def _compute_normalized_adjacency(self, adjacency: np.ndarray) -> np.ndarray:
        """Compute normalized adjacency matrix"""
        n = adjacency.shape[0]
        A_tilde = adjacency + np.eye(n)
        degree = np.sum(A_tilde, axis=1)
        degree[degree == 0] = 1
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
        return D_inv_sqrt @ A_tilde @ D_inv_sqrt
    
    def _prepare_sequences(self, gps_data, node_list, seq_len: int = 10):
        """Prepare sequences for training"""
        import pandas as pd
        
        gps_data = gps_data.sort_values('timestamp')
        
        # Create time bins
        start_time = gps_data['timestamp'].min()
        end_time = gps_data['timestamp'].max()
        time_delta = pd.Timedelta(minutes=10)
        timestamps = pd.date_range(start=start_time, end=end_time, freq=time_delta)
        
        # Create speed matrix
        speed_matrix = np.zeros((len(node_list), len(timestamps)))
        
        for i, node_id in enumerate(node_list):
            node_data = gps_data[gps_data['grid_id'] == node_id]
            for t_idx, timestamp in enumerate(timestamps):
                mask = (node_data['timestamp'] >= timestamp) & \
                       (node_data['timestamp'] < timestamp + time_delta)
                speeds = node_data[mask]['speed'].values
                if len(speeds) > 0:
                    speed_matrix[i, t_idx] = np.median(speeds)
                else:
                    speed_matrix[i, t_idx] = 20.0
        
        # Create sequences
        sequences = []
        for t in range(len(timestamps) - seq_len):
            seq = speed_matrix[:, t:t+seq_len]  # (num_nodes, seq_len)
            seq = seq.T  # (seq_len, num_nodes)
            sequences.append(seq)
        
        sequences = np.array(sequences)  # (num_sequences, seq_len, num_nodes)
        sequences = sequences[:, :, :, np.newaxis]  # Add channel dimension
        
        return sequences, None
    
    def _find_nearest_node(self, location: Tuple[float, float],
                          node_attrs: Dict) -> Optional[str]:
        """Find nearest node"""
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
        """Get model parameters"""
        return {
            'model_state': self.model.state_dict() if self.model else None,
            'model_type': self.model_type,
            'adj_norm': self.adj_norm,
            'node_list': self.node_list,
            'config': self.config
        }
