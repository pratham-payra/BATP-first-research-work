"""
Data Preprocessing Module
Implements Algorithm 1: Data Preprocessing section
Handles grid construction, graph building, MST extraction, and adjacency matrix creation
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import cdist
from ..utils.haversine import haversine_distance
from .s3_manager import get_s3_manager


class DataPreprocessor:
    """Handles data preprocessing for graph construction"""
    
    def __init__(self, route: str, grid_size: float = 50.0):
        """
        Initialize preprocessor
        
        Args:
            route: Route identifier
            grid_size: Grid cell size in meters (default 50m x 50m)
        """
        self.route = route
        self.grid_size = grid_size
        self.s3_manager = get_s3_manager()
        
        # Approximate degrees per meter (at equator)
        # 1 degree latitude ≈ 111,000 meters
        # 1 degree longitude ≈ 111,000 * cos(latitude) meters
        self.meters_per_degree_lat = 111000
        self.grid_size_degrees = grid_size / self.meters_per_degree_lat
    
    def compute_speeds(self, gps_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute speeds between consecutive GPS points using Haversine distance
        
        Args:
            gps_data: DataFrame with latitude, longitude, timestamp
        
        Returns:
            DataFrame with added 'speed' column (km/h)
        """
        print("Computing speeds from GPS data...")
        
        gps_data = gps_data.sort_values(['session_id', 'timestamp']).reset_index(drop=True)
        
        speeds = []
        
        for session_id in gps_data['session_id'].unique():
            session_data = gps_data[gps_data['session_id'] == session_id].copy()
            session_speeds = [0]  # First point has no speed
            
            for i in range(1, len(session_data)):
                lat1, lon1 = session_data.iloc[i-1][['latitude', 'longitude']]
                lat2, lon2 = session_data.iloc[i][['latitude', 'longitude']]
                t1 = session_data.iloc[i-1]['timestamp']
                t2 = session_data.iloc[i]['timestamp']
                
                # Compute Haversine distance in km
                distance = haversine_distance(lat1, lon1, lat2, lon2)
                
                # Compute time difference in hours
                time_diff = (t2 - t1).total_seconds() / 3600.0
                
                # Compute speed in km/h
                if time_diff > 0:
                    speed = distance / time_diff
                else:
                    speed = 0
                
                session_speeds.append(speed)
            
            speeds.extend(session_speeds)
        
        gps_data['speed'] = speeds
        
        # Filter out unrealistic speeds (e.g., > 100 km/h for buses)
        gps_data.loc[gps_data['speed'] > 100, 'speed'] = gps_data['speed'].median()
        
        print(f"Computed speeds: mean={gps_data['speed'].mean():.2f} km/h, "
              f"median={gps_data['speed'].median():.2f} km/h")
        
        return gps_data
    
    def build_grid(self, gps_data: pd.DataFrame) -> pd.DataFrame:
        """
        Build grid cells by rounding GPS coordinates
        
        Args:
            gps_data: DataFrame with latitude, longitude
        
        Returns:
            DataFrame with added grid_lat, grid_lon columns
        """
        print(f"Building {self.grid_size}m x {self.grid_size}m grid...")
        
        # Round coordinates to grid
        gps_data['grid_lat'] = np.round(gps_data['latitude'] / self.grid_size_degrees) * self.grid_size_degrees
        gps_data['grid_lon'] = np.round(gps_data['longitude'] / self.grid_size_degrees) * self.grid_size_degrees
        
        # Create grid cell ID
        gps_data['grid_id'] = (gps_data['grid_lat'].astype(str) + '_' + 
                               gps_data['grid_lon'].astype(str))
        
        num_cells = gps_data['grid_id'].nunique()
        print(f"Created {num_cells} grid cells")
        
        return gps_data
    
    def compute_grid_medians(self, gps_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute median coordinates and speeds for each grid cell
        
        Args:
            gps_data: DataFrame with grid_id, latitude, longitude, speed
        
        Returns:
            DataFrame with grid cell statistics
        """
        print("Computing grid cell medians...")
        
        grid_stats = gps_data.groupby('grid_id').agg({
            'latitude': 'median',
            'longitude': 'median',
            'grid_lat': 'first',
            'grid_lon': 'first',
            'speed': ['median', 'mean', 'std', 'count']
        }).reset_index()
        
        # Flatten column names
        grid_stats.columns = ['grid_id', 'median_lat', 'median_lon', 'grid_lat', 
                             'grid_lon', 'median_speed', 'mean_speed', 'std_speed', 'count']
        
        print(f"Computed statistics for {len(grid_stats)} grid cells")
        
        return grid_stats
    
    def build_graph(self, grid_stats: pd.DataFrame, gps_data: pd.DataFrame) -> Tuple[nx.Graph, Dict]:
        """
        Build graph from grid cells with edges based on GPS transitions
        
        Args:
            grid_stats: DataFrame with grid cell statistics
            gps_data: Original GPS data with grid_id
        
        Returns:
            Tuple of (NetworkX graph, node attributes dict)
        """
        print("Building graph from grid cells...")
        
        G = nx.Graph()
        
        # Add nodes with attributes
        node_attrs = {}
        for idx, row in grid_stats.iterrows():
            node_id = row['grid_id']
            G.add_node(node_id)
            node_attrs[node_id] = {
                'median_lat': row['median_lat'],
                'median_lon': row['median_lon'],
                'grid_lat': row['grid_lat'],
                'grid_lon': row['grid_lon'],
                'median_speed': row['median_speed'],
                'mean_speed': row['mean_speed'],
                'node_index': idx
            }
        
        nx.set_node_attributes(G, node_attrs)
        
        # Add edges based on GPS transitions
        gps_data = gps_data.sort_values(['session_id', 'timestamp'])
        
        edge_counts = {}
        for session_id in gps_data['session_id'].unique():
            session_data = gps_data[gps_data['session_id'] == session_id]
            
            for i in range(len(session_data) - 1):
                node1 = session_data.iloc[i]['grid_id']
                node2 = session_data.iloc[i+1]['grid_id']
                
                if node1 != node2:  # Only add edges between different cells
                    edge = tuple(sorted([node1, node2]))
                    edge_counts[edge] = edge_counts.get(edge, 0) + 1
        
        # Add edges with Haversine distances
        for (node1, node2), count in edge_counts.items():
            lat1, lon1 = node_attrs[node1]['median_lat'], node_attrs[node1]['median_lon']
            lat2, lon2 = node_attrs[node2]['median_lat'], node_attrs[node2]['median_lon']
            
            distance = haversine_distance(lat1, lon1, lat2, lon2)
            
            G.add_edge(node1, node2, weight=distance, count=count)
        
        print(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        return G, node_attrs
    
    def extract_mst(self, G: nx.Graph) -> nx.Graph:
        """
        Extract Minimum Spanning Tree from graph
        
        Args:
            G: NetworkX graph
        
        Returns:
            MST as NetworkX graph
        """
        print("Extracting Minimum Spanning Tree...")
        
        # Use Kruskal's algorithm (default in NetworkX)
        mst = nx.minimum_spanning_tree(G, weight='weight')
        
        print(f"MST has {mst.number_of_nodes()} nodes and {mst.number_of_edges()} edges")
        
        return mst
    
    def compute_shortest_path(self, G: nx.Graph, start_node: str, 
                             end_node: str) -> List[str]:
        """
        Compute shortest path between two nodes using Dijkstra's algorithm
        
        Args:
            G: NetworkX graph
            start_node: Starting node ID
            end_node: Ending node ID
        
        Returns:
            List of node IDs in shortest path
        """
        try:
            path = nx.shortest_path(G, start_node, end_node, weight='weight')
            return path
        except nx.NetworkXNoPath:
            print(f"No path found between {start_node} and {end_node}")
            return []
    
    def build_adjacency_matrix(self, G: nx.Graph, node_attrs: Dict) -> np.ndarray:
        """
        Build adjacency matrix with Haversine distances
        
        Args:
            G: NetworkX graph
            node_attrs: Node attributes dictionary
        
        Returns:
            Adjacency matrix (n x n)
        """
        print("Building adjacency matrix...")
        
        n = G.number_of_nodes()
        A = np.zeros((n, n))
        
        node_list = list(G.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}
        
        for node1, node2, data in G.edges(data=True):
            idx1 = node_to_idx[node1]
            idx2 = node_to_idx[node2]
            distance = data['weight']
            
            A[idx1, idx2] = distance
            A[idx2, idx1] = distance  # Symmetric
        
        print(f"Built adjacency matrix of shape {A.shape}")
        
        return A
    
    def preprocess_full_pipeline(self, gps_data: pd.DataFrame, 
                                 save_to_s3: bool = True) -> Dict:
        """
        Complete preprocessing pipeline
        
        Args:
            gps_data: Raw GPS data with weather
            save_to_s3: Whether to save preprocessed data to S3
        
        Returns:
            Dictionary with all preprocessed components
        """
        print("=" * 60)
        print("Starting full preprocessing pipeline...")
        print("=" * 60)
        
        # Step 1: Compute speeds
        gps_data = self.compute_speeds(gps_data)
        
        # Step 2: Build grid
        gps_data = self.build_grid(gps_data)
        
        # Step 3: Compute grid statistics
        grid_stats = self.compute_grid_medians(gps_data)
        
        # Step 4: Build graph
        graph, node_attrs = self.build_graph(grid_stats, gps_data)
        
        # Step 5: Extract MST
        mst = self.extract_mst(graph)
        
        # Step 6: Build adjacency matrix
        adjacency_matrix = self.build_adjacency_matrix(graph, node_attrs)
        
        # Package results
        preprocessed_data = {
            'gps_data': gps_data,
            'grid_stats': grid_stats,
            'graph': graph,
            'mst': mst,
            'node_attrs': node_attrs,
            'adjacency_matrix': adjacency_matrix,
            'node_list': list(graph.nodes()),
            'metadata': {
                'num_nodes': graph.number_of_nodes(),
                'num_edges': graph.number_of_edges(),
                'num_mst_edges': mst.number_of_edges(),
                'grid_size_meters': self.grid_size,
                'grid_size_degrees': self.grid_size_degrees
            }
        }
        
        # Save to S3 if requested
        if save_to_s3:
            print("\nSaving preprocessed data to S3...")
            self.s3_manager.save_preprocessed_data(
                preprocessed_data, 
                self.route, 
                'graph_data'
            )
        
        print("=" * 60)
        print("Preprocessing pipeline complete!")
        print("=" * 60)
        
        return preprocessed_data
    
    def load_preprocessed_data(self) -> Dict:
        """Load preprocessed data from S3"""
        print(f"Loading preprocessed data for {self.route} from S3...")
        return self.s3_manager.load_preprocessed_data(self.route, 'graph_data')
