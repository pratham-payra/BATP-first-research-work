"""
Training script for MST-AV model
"""

from ..models.mst_av import MSTAV
from ..data.s3_manager import get_s3_manager


def train_mst_av(preprocessed_data: dict, route: str):
    """
    Train MST-AV model
    
    Args:
        preprocessed_data: Dictionary with preprocessed components
        route: Route identifier
    
    Returns:
        Trained MSTAV model
    """
    print("\n" + "="*60)
    print("Training MST-AV Model")
    print("="*60)
    
    # Extract data
    gps_data = preprocessed_data['gps_data']
    graph = preprocessed_data['graph']
    mst = preprocessed_data['mst']
    node_attrs = preprocessed_data['node_attrs']
    
    # Initialize and train model
    model = MSTAV()
    model.train(gps_data, graph, mst, node_attrs)
    
    # Save model to S3
    s3_manager = get_s3_manager()
    model_params = model.get_model_params()
    
    metadata = {
        'model_type': 'MST-AV',
        'route': route,
        'num_nodes': len(node_attrs),
        'training_samples': len(gps_data)
    }
    
    s3_manager.save_model(model_params, 'mst_av', route, metadata)
    
    print("MST-AV model saved to S3")
    print("="*60 + "\n")
    
    return model
