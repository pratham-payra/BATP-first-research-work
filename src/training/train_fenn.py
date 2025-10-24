"""
Training script for FE-NN (Feature-Encoded Neural Network) model
"""

from ..models.fenn import FENN
from ..data.s3_manager import get_s3_manager
import yaml


def train_fenn(preprocessed_data: dict, route: str):
    """
    Train FE-NN model
    
    Args:
        preprocessed_data: Dictionary with preprocessed components
        route: Route identifier
    
    Returns:
        Trained FENN model
    """
    print("\n" + "="*60)
    print("Training FE-NN Model")
    print("="*60)
    
    # Load config
    with open('config/model_config.yaml', 'r') as f:
        config = yaml.safe_load(f)['fenn']
    
    # Extract data
    gps_data = preprocessed_data['gps_data']
    
    # Initialize and train model
    model = FENN(config)
    model.train(gps_data)
    
    # Save model to S3
    s3_manager = get_s3_manager()
    model_params = model.get_model_params()
    
    metadata = {
        'model_type': 'FE-NN',
        'route': route,
        'config': config
    }
    
    s3_manager.save_model(model_params, 'fenn', route, metadata)
    
    print("FE-NN model saved to S3")
    print("="*60 + "\n")
    
    return model
