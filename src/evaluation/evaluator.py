"""
Evaluation Pipeline
Evaluates trained models and stores results to S3
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from .metrics import calculate_all_metrics, categorize_eta_ranges
from .results_manager import ResultsManager
from ..data.s3_manager import get_s3_manager


class ModelEvaluator:
    """Evaluates models and stores results to S3"""
    
    def __init__(self, route: str):
        """
        Initialize evaluator
        
        Args:
            route: Route identifier
        """
        self.route = route
        self.results_manager = ResultsManager(route)
        self.s3_manager = get_s3_manager()
    
    def evaluate_model(self, model, model_name: str, test_data: pd.DataFrame,
                      graph, node_attrs, bus_stops: pd.DataFrame) -> Dict:
        """
        Evaluate a single model on test data
        
        Args:
            model: Trained model
            model_name: Name of the model
            test_data: Test GPS data
            graph: NetworkX graph
            node_attrs: Node attributes
            bus_stops: Bus stop locations
        
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\nEvaluating {model_name}...")
        
        predictions = []
        actuals = []
        
        # Sample test queries
        num_samples = min(500, len(bus_stops) * 10)
        
        for i in range(num_samples):
            # Sample a bus location and target stop
            bus_idx = np.random.randint(0, len(test_data))
            stop_idx = np.random.randint(0, len(bus_stops))
            
            bus_location = (
                test_data.iloc[bus_idx]['latitude'],
                test_data.iloc[bus_idx]['longitude']
            )
            
            stop_location = (
                bus_stops.iloc[stop_idx]['Latitude'],
                bus_stops.iloc[stop_idx]['Longitude']
            )
            
            try:
                # Get prediction
                if model_name == 'mst_av':
                    eta_pred = model.predict(bus_location, stop_location)
                elif model_name in ['gdrn_dft', 'knn', 'fenn', 'mgcn']:
                    eta_pred = model.predict(
                        bus_location, stop_location,
                        graph=graph, node_attrs=node_attrs
                    )
                else:
                    eta_pred = 0.0
                
                # Compute actual ETA (simplified - using distance and average speed)
                from ..utils.haversine import haversine_distance
                distance = haversine_distance(
                    bus_location[0], bus_location[1],
                    stop_location[0], stop_location[1]
                )
                avg_speed = test_data['speed'].median()
                eta_actual = (distance / avg_speed) * 60.0 if avg_speed > 0 else 0.0
                
                predictions.append(eta_pred)
                actuals.append(eta_actual)
                
            except Exception as e:
                print(f"Error in prediction {i}: {e}")
                continue
        
        # Calculate metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Overall metrics
        overall_metrics = calculate_all_metrics(actuals, predictions)
        
        # Metrics by ETA range
        eta_categories = categorize_eta_ranges(actuals)
        range_metrics = {}
        
        for category in ['0-10', '10-25', '25-45', '45+']:
            mask = eta_categories == category
            if np.sum(mask) > 0:
                range_metrics[category] = calculate_all_metrics(
                    actuals[mask], predictions[mask]
                )
        
        results = {
            'model': model_name,
            'overall': overall_metrics,
            'by_range': range_metrics,
            'num_samples': len(predictions)
        }
        
        print(f"{model_name} - MAE: {overall_metrics['MAE']:.2f}, "
              f"MAPE: {overall_metrics['MAPE']:.2f}%, "
              f"RMSE: {overall_metrics['RMSE']:.2f}")
        
        return results
    
    def evaluate_all_models(self, models: Dict, test_data: pd.DataFrame,
                           graph, node_attrs, bus_stops: pd.DataFrame):
        """
        Evaluate all models and save results to S3
        
        Args:
            models: Dictionary of trained models
            test_data: Test GPS data
            graph: NetworkX graph
            node_attrs: Node attributes
            bus_stops: Bus stop locations
        """
        print("\n" + "="*80)
        print("EVALUATING ALL MODELS")
        print("="*80)
        
        all_results = {}
        
        for model_name, model in models.items():
            results = self.evaluate_model(
                model, model_name, test_data, graph, node_attrs, bus_stops
            )
            all_results[model_name] = results
        
        # Format and save results
        self._save_results(all_results)
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETE - Results saved to S3")
        print("="*80)
    
    def _save_results(self, all_results: Dict):
        """Format and save results to S3"""
        
        # Format for model performance table
        performance_dict = {}
        
        for model_name, results in all_results.items():
            performance_dict[model_name] = {
                'MAE': {},
                'MAPE': {},
                'RMSE': {}
            }
            
            # Overall metrics
            for metric in ['MAE', 'MAPE', 'RMSE']:
                performance_dict[model_name][metric]['Overall'] = \
                    results['overall'][metric]
            
            # Range metrics
            for range_name, metrics in results.get('by_range', {}).items():
                for metric in ['MAE', 'MAPE', 'RMSE']:
                    performance_dict[model_name][metric][range_name] = \
                        metrics[metric]
        
        # Save to S3
        self.results_manager.save_model_performance(performance_dict)
        
        print(f"\nResults saved to S3 bucket: {self.s3_manager.results_bucket}")
        print(f"Path: model_performance/{self.route}_performance.csv")
