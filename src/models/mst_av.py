"""
MST-AV Model: Minimum Spanning Tree with Average Velocities
Implements Algorithms 2 and 3 from the paper
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from ..utils.haversine import haversine_distance


class MSTAV:
    """
    MST-AV Model for bus arrival time estimation
    Uses historical mean speeds on MST-augmented graph
    """
    
    def __init__(self):
        """Initialize MST-AV model"""
        self.mean_speeds = {}  # Dictionary: node_id -> mean speed
        self.graph = None
        self.mst = None
        self.node_attrs = None
        
    def train(self, gps_data, graph, mst, node_attrs):
        """
        Training Algorithm for MST-AV (Algorithm 2)
        Compute historical mean speeds for all nodes
        
        Args:
            gps_data: DataFrame with grid_id and speed columns
            graph: NetworkX graph
            mst: Minimum spanning tree
            node_attrs: Node attributes dictionary
        """
        print("Training MST-AV model...")
        
        self.graph = graph
        self.mst = mst
        self.node_attrs = node_attrs
        
        # Initialize sum and count for each node
        speed_sum = {}
        speed_count = {}
        
        for node in graph.nodes():
            speed_sum[node] = 0.0
            speed_count[node] = 0
        
        # Compute sum and count from historical traces
        for idx, row in gps_data.iterrows():
            node_id = row['grid_id']
            speed = row['speed']
            
            if node_id in speed_sum and speed > 0:  # Valid speed
                speed_sum[node_id] += speed
                speed_count[node_id] += 1
        
        # Compute mean speeds
        global_speeds = []
        for node in graph.nodes():
            if speed_count[node] > 0:
                self.mean_speeds[node] = speed_sum[node] / speed_count[node]
                global_speeds.append(self.mean_speeds[node])
            else:
                # Use default value (will be set to global average)
                self.mean_speeds[node] = None
        
        # Set default for nodes without data
        global_avg = np.mean(global_speeds) if global_speeds else 20.0  # 20 km/h default
        
        for node in self.mean_speeds:
            if self.mean_speeds[node] is None:
                self.mean_speeds[node] = global_avg
        
        print(f"Computed mean speeds for {len(self.mean_speeds)} nodes")
        print(f"Global average speed: {global_avg:.2f} km/h")
        
        return self
    
    def predict(self, bus_location: Tuple[float, float], 
                stop_location: Tuple[float, float]) -> float:
        """
        Computation Algorithm for MST-AV (Algorithm 3)
        Estimate bus arrival time
        
        Args:
            bus_location: (latitude, longitude) of current bus location
            stop_location: (latitude, longitude) of bus stop
        
        Returns:
            Estimated Time of Arrival in minutes
        """
        # Step 1: Identify grid cells for bus and stop
        bus_node = self._find_nearest_node(bus_location)
        stop_node = self._find_nearest_node(stop_location)
        
        if bus_node is None or stop_node is None:
            print("Warning: Could not find nodes for given locations")
            return 0.0
        
        # Step 2: Compute shortest path on MST-augmented graph
        try:
            path = nx.shortest_path(self.graph, bus_node, stop_node, weight='weight')
        except nx.NetworkXNoPath:
            print(f"Warning: No path found between {bus_node} and {stop_node}")
            return 0.0
        
        # Step 3: Compute ETA by summing edge travel times
        eta = 0.0
        
        for i in range(len(path) - 1):
            node_i = path[i]
            node_j = path[i + 1]
            
            # Get node coordinates
            lat_i = self.node_attrs[node_i]['median_lat']
            lon_i = self.node_attrs[node_i]['median_lon']
            lat_j = self.node_attrs[node_j]['median_lat']
            lon_j = self.node_attrs[node_j]['median_lon']
            
            # Compute edge length (Haversine distance in km)
            edge_length = haversine_distance(lat_i, lon_i, lat_j, lon_j)
            
            # Compute average velocity for edge
            speed_i = self.mean_speeds.get(node_i, 20.0)
            speed_j = self.mean_speeds.get(node_j, 20.0)
            avg_speed = (speed_i + speed_j) / 2.0
            
            # Avoid division by zero
            if avg_speed < 0.1:
                avg_speed = 20.0
            
            # Compute travel time in hours, convert to minutes
            travel_time = (edge_length / avg_speed) * 60.0
            eta += travel_time
        
        return eta
    
    def _find_nearest_node(self, location: Tuple[float, float]) -> Optional[str]:
        """
        Find nearest grid node to given location
        
        Args:
            location: (latitude, longitude)
        
        Returns:
            Node ID of nearest node
        """
        lat, lon = location
        min_distance = float('inf')
        nearest_node = None
        
        for node_id, attrs in self.node_attrs.items():
            node_lat = attrs['median_lat']
            node_lon = attrs['median_lon']
            
            distance = haversine_distance(lat, lon, node_lat, node_lon)
            
            if distance < min_distance:
                min_distance = distance
                nearest_node = node_id
        
        return nearest_node
    
    def get_model_params(self) -> Dict:
        """Get model parameters for saving"""
        return {
            'mean_speeds': self.mean_speeds,
            'node_attrs': self.node_attrs
        }
    
    def set_model_params(self, params: Dict):
        """Set model parameters from loaded data"""
        self.mean_speeds = params['mean_speeds']
        self.node_attrs = params['node_attrs']
