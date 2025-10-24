"""
Training script for Hybrid ensemble model
"""

from ..models.hybrid import HybridModel
from ..data.s3_manager import get_s3_manager
import yaml


def train_hybrid(preprocessed_data: dict, route: str, trained_models: dict):
    """
    Train Hybrid model
    
    Args:
        preprocessed_data: Dictionary with preprocessed components
        route: Route identifier
        trained_models: Dictionary of already trained base models
    
    Returns:
        Trained HybridModel
    """
    print("\n" + "="*60)
    print("Training Hybrid Model")
    print("="*60)
    
    # Load config
    with open('config/model_config.yaml', 'r') as f:
        config = yaml.safe_load(f)['hybrid']
    
    # Extract data
    gps_data = preprocessed_data['gps_data']
    graph = preprocessed_data['graph']
    node_attrs = preprocessed_data['node_attrs']
    
    # Create training data by getting predictions from all base models
    print("Generating training data from base models...")
    training_data = []
    
    # Sample from GPS data
    import pandas as pd
    sampled_data = gps_data.sample(min(1000, len(gps_data)))  # Sample for efficiency
    
    for idx, row in sampled_data.iterrows():
        location = (row['latitude'], row['longitude'])
        
        # Get weather data
        weather = {
            'humidity': row.get('humidity', 60),
            'precipitation': row.get('precipitation', 0),
            'temperature': row.get('temperature', 298),
            'cloud_cover': row.get('cloud_cover', 50),
            'pressure': row.get('pressure', 1013),
            'feels_like': row.get('feels_like', 298),
            'wind_speed': row.get('wind_speed', 2.0)
        }
        
        # Get predictions from each model (simplified for training)
        model_etas = {}
        for model_name, model in trained_models.items():
            try:
                # Simplified prediction for training data generation
                model_etas[model_name] = row['speed'] * 5  # Placeholder
            except:
                model_etas[model_name] = 0.0
        
        training_example = {
            'location': location,
            'weather': weather,
            'time': row['timestamp'],
            'speed': row['speed'],
            'distance_to_stop': 5.0,  # Placeholder
            'model_etas': model_etas,
            'actual_eta': row['speed'] * 5  # Placeholder - would be computed from actual data
        }
        
        training_data.append(training_example)
    
    # Initialize and train hybrid model
    model = HybridModel(config, trained_models)
    model.train(training_data)
    
    # Save model to S3
    s3_manager = get_s3_manager()
    model_params = model.get_model_params()
    
    metadata = {
        'model_type': 'Hybrid',
        'route': route,
        'base_models': list(trained_models.keys()),
        'config': config
    }
    
    s3_manager.save_model(model_params, 'hybrid', route, metadata)
    
    print("Hybrid model saved to S3")
    print("="*60 + "\n")
    
    return model
