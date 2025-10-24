"""
Training script for MGCN (Masked Graph Convolutional Network) model
"""

from ..models.mgcn import MGCN
from ..data.s3_manager import get_s3_manager
import yaml


def train_mgcn(preprocessed_data: dict, route: str):
    """
    Train MGCN model
    
    Args:
        preprocessed_data: Dictionary with preprocessed components
        route: Route identifier
    
    Returns:
        Trained MGCN model
    """
    print("\n" + "="*60)
    print("Training MGCN Model")
    print("="*60)
    
    # Load config
    with open('config/model_config.yaml', 'r') as f:
        config = yaml.safe_load(f)['mgcn']
    
    # Extract data
    gps_data = preprocessed_data['gps_data']
    adjacency_matrix = preprocessed_data['adjacency_matrix']
    node_list = preprocessed_data['node_list']
    
    # Load DFT speeds if available (from GDRN-DFT)
    s3_manager = get_s3_manager()
    try:
        dft_coeffs = s3_manager.load_dft_coefficients(route)
        # Could reconstruct DFT speeds here if needed
        dft_speeds = None
    except:
        dft_speeds = None
    
    # Initialize and train model
    model = MGCN(config)
    model.train(gps_data, adjacency_matrix, node_list, dft_speeds)
    
    # Save model to S3
    model_params = model.get_model_params()
    
    metadata = {
        'model_type': 'MGCN',
        'route': route,
        'num_nodes': len(node_list),
        'config': config
    }
    
    s3_manager.save_model(model_params, 'mgcn', route, metadata)
    
    print("MGCN model saved to S3")
    print("="*60 + "\n")
    
    return model
