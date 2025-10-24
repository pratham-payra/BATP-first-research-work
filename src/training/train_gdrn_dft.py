"""
Training script for GDRN-DFT model
"""

from ..models.gdrn_dft import GDRNDFT
from ..data.s3_manager import get_s3_manager
import yaml


def train_gdrn_dft(preprocessed_data: dict, route: str):
    """
    Train GDRN-DFT model
    
    Args:
        preprocessed_data: Dictionary with preprocessed components
        route: Route identifier
    
    Returns:
        Trained GDRNDFT model
    """
    print("\n" + "="*60)
    print("Training GDRN-DFT Model")
    print("="*60)
    
    # Load config
    with open('config/model_config.yaml', 'r') as f:
        config = yaml.safe_load(f)['gdrn_dft']
    
    # Extract data
    gps_data = preprocessed_data['gps_data']
    adjacency_matrix = preprocessed_data['adjacency_matrix']
    node_list = preprocessed_data['node_list']
    
    # Initialize and train model
    model = GDRNDFT(config)
    model.train(gps_data, adjacency_matrix, node_list)
    
    # Save model and DFT coefficients to S3
    s3_manager = get_s3_manager()
    model_params = model.get_model_params()
    
    metadata = {
        'model_type': 'GDRN-DFT',
        'route': route,
        'num_nodes': len(node_list),
        'T': model.T,
        'config': config
    }
    
    s3_manager.save_model(model_params, 'gdrn_dft', route, metadata)
    
    # Save DFT coefficients separately
    dft_coeffs = {
        'amplitudes': model.dft_amplitudes,
        'phases': model.dft_phases
    }
    s3_manager.save_dft_coefficients(dft_coeffs, route)
    
    print("GDRN-DFT model and DFT coefficients saved to S3")
    print("="*60 + "\n")
    
    return model
