"""
Training script for KNN (Koopman Neural Network) model
"""

from ..models.knn import KNN
from ..data.s3_manager import get_s3_manager
import yaml


def train_knn(preprocessed_data: dict, route: str):
    """
    Train KNN model
    
    Args:
        preprocessed_data: Dictionary with preprocessed components
        route: Route identifier
    
    Returns:
        Trained KNN model
    """
    print("\n" + "="*60)
    print("Training KNN Model")
    print("="*60)
    
    # Load config
    with open('config/model_config.yaml', 'r') as f:
        config = yaml.safe_load(f)['knn']
    
    # Extract data
    gps_data = preprocessed_data['gps_data']
    
    # Initialize and train model
    model = KNN(config)
    model.train(gps_data)
    
    # Save model to S3
    s3_manager = get_s3_manager()
    model_params = model.get_model_params()
    
    metadata = {
        'model_type': 'KNN',
        'route': route,
        'config': config
    }
    
    s3_manager.save_model(model_params, 'knn', route, metadata)
    
    print("KNN model saved to S3")
    print("="*60 + "\n")
    
    return model
