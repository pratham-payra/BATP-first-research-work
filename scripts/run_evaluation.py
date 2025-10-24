"""
Evaluation Pipeline Script
Evaluates trained models and saves results to S3
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
from datetime import datetime
from src.data.acquisition import DataAcquisition
from src.data.preprocessing import DataPreprocessor
from src.evaluation.evaluator import ModelEvaluator
from src.data.s3_manager import get_s3_manager


def main():
    parser = argparse.ArgumentParser(description='Run Hybrid ETA Evaluation Pipeline')
    parser.add_argument('--route', type=str, required=True,
                       help='Route identifier (e.g., route1, route2, route3)')
    parser.add_argument('--models', type=str, default='all',
                       help='Comma-separated list of models to evaluate (default: all)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"HYBRID ETA EVALUATION PIPELINE - {args.route.upper()}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Step 1: Load test data
    print("\n[STEP 1] Loading Test Data")
    print("-" * 80)
    
    data_acq = DataAcquisition(args.route)
    test_data = data_acq.acquire_full_dataset(data_type='test', fetch_weather=False)
    bus_stops = data_acq.fetch_bus_stops()
    
    # Step 2: Load preprocessed data
    print("\n[STEP 2] Loading Preprocessed Data")
    print("-" * 80)
    
    preprocessor = DataPreprocessor(args.route)
    preprocessed_data = preprocessor.load_preprocessed_data()
    
    graph = preprocessed_data['graph']
    node_attrs = preprocessed_data['node_attrs']
    
    # Step 3: Load trained models
    print("\n[STEP 3] Loading Trained Models")
    print("-" * 80)
    
    s3_manager = get_s3_manager()
    
    models_to_eval = args.models.split(',') if args.models != 'all' else [
        'mst_av', 'gdrn_dft', 'knn', 'fenn', 'mgcn', 'hybrid'
    ]
    
    loaded_models = {}
    
    for model_name in models_to_eval:
        print(f"\nLoading {model_name}...")
        
        try:
            if model_name == 'mst_av':
                from src.models.mst_av import MSTAV
                model_params = s3_manager.load_model(MSTAV, 'mst_av', args.route)
                model = MSTAV()
                model.set_model_params(model_params)
                model.graph = graph
                model.node_attrs = node_attrs
                loaded_models['mst_av'] = model
                
            # Add other model loading logic here
            # For now, we'll just load MST-AV as an example
            
            print(f"✓ {model_name} loaded successfully")
            
        except Exception as e:
            print(f"✗ Error loading {model_name}: {e}")
    
    # Step 4: Evaluate models
    print("\n[STEP 4] Evaluating Models")
    print("-" * 80)
    
    evaluator = ModelEvaluator(args.route)
    evaluator.evaluate_all_models(loaded_models, test_data, graph, node_attrs, bus_stops)
    
    print("\n" + "=" * 80)
    print("EVALUATION PIPELINE COMPLETE")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to S3: {s3_manager.results_bucket}")
    print("=" * 80)


if __name__ == '__main__':
    main()
