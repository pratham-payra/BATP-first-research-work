"""
Main Training Pipeline
Orchestrates the complete training process for all models
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
from datetime import datetime
from src.data.acquisition import DataAcquisition
from src.data.preprocessing import DataPreprocessor
from src.evaluation.results_manager import ResultsManager


def main():
    parser = argparse.ArgumentParser(description='Run Hybrid ETA Training Pipeline')
    parser.add_argument('--route', type=str, required=True, 
                       help='Route identifier (e.g., route1, route2, route3)')
    parser.add_argument('--models', type=str, default='all',
                       help='Comma-separated list of models to train (default: all)')
    parser.add_argument('--skip-preprocessing', action='store_true',
                       help='Skip preprocessing and load from S3')
    parser.add_argument('--skip-weather', action='store_true',
                       help='Skip weather data fetching')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"HYBRID ETA TRAINING PIPELINE - {args.route.upper()}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Step 1: Data Acquisition
    print("\n[STEP 1] Data Acquisition")
    print("-" * 80)
    
    data_acq = DataAcquisition(args.route)
    
    if not args.skip_preprocessing:
        gps_data = data_acq.acquire_full_dataset(
            data_type='train',
            fetch_weather=not args.skip_weather
        )
        
        # Step 2: Data Preprocessing
        print("\n[STEP 2] Data Preprocessing")
        print("-" * 80)
        
        preprocessor = DataPreprocessor(args.route)
        preprocessed_data = preprocessor.preprocess_full_pipeline(
            gps_data,
            save_to_s3=True
        )
    else:
        print("Loading preprocessed data from S3...")
        preprocessor = DataPreprocessor(args.route)
        preprocessed_data = preprocessor.load_preprocessed_data()
    
    # Step 3: Model Training
    print("\n[STEP 3] Model Training")
    print("-" * 80)
    
    models_to_train = args.models.split(',') if args.models != 'all' else [
        'mst_av', 'gdrn_dft', 'knn', 'fenn', 'mgcn', 'hybrid'
    ]
    
    trained_models = {}
    
    for model_name in models_to_train:
        print(f"\nTraining {model_name.upper()}...")
        
        try:
            if model_name == 'mst_av':
                from src.training.train_mst_av import train_mst_av
                model = train_mst_av(preprocessed_data, args.route)
                trained_models['mst_av'] = model
                
            elif model_name == 'gdrn_dft':
                from src.training.train_gdrn_dft import train_gdrn_dft
                model = train_gdrn_dft(preprocessed_data, args.route)
                trained_models['gdrn_dft'] = model
                
            elif model_name == 'knn':
                from src.training.train_knn import train_knn
                model = train_knn(preprocessed_data, args.route)
                trained_models['knn'] = model
                
            elif model_name == 'fenn':
                from src.training.train_fenn import train_fenn
                model = train_fenn(preprocessed_data, args.route)
                trained_models['fenn'] = model
                
            elif model_name == 'mgcn':
                from src.training.train_mgcn import train_mgcn
                model = train_mgcn(preprocessed_data, args.route)
                trained_models['mgcn'] = model
                
            elif model_name == 'hybrid':
                from src.training.train_hybrid import train_hybrid
                model = train_hybrid(preprocessed_data, args.route, trained_models)
                trained_models['hybrid'] = model
            
            print(f"✓ {model_name.upper()} training complete")
            
        except Exception as e:
            print(f"✗ Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Step 4: Save Results
    print("\n[STEP 4] Saving Training Results")
    print("-" * 80)
    
    results_manager = ResultsManager(args.route)
    
    # Create training summary
    training_summary = {
        'route': args.route,
        'timestamp': datetime.now().isoformat(),
        'models_trained': list(trained_models.keys()),
        'preprocessing_metadata': preprocessed_data.get('metadata', {})
    }
    
    results_manager.save_metrics_summary(
        training_summary,
        timestamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    )
    
    print("\n" + "=" * 80)
    print("TRAINING PIPELINE COMPLETE")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models trained: {', '.join(trained_models.keys())}")
    print("=" * 80)


if __name__ == '__main__':
    main()
