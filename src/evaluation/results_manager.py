"""
Results Manager
Handles storage and retrieval of evaluation results to/from S3
Formats results according to the provided CSV structures
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from ..data.s3_manager import get_s3_manager


class ResultsManager:
    """Manages evaluation results storage to S3"""
    
    def __init__(self, route: str):
        """
        Initialize results manager
        
        Args:
            route: Route identifier (e.g., 'route1', 'route2', 'route3')
        """
        self.route = route
        self.s3_manager = get_s3_manager()
    
    def save_model_performance(self, results: Dict[str, Dict[str, float]], 
                               conditions: List[str] = None):
        """
        Save model performance results (like Table 1 in your results)
        
        Args:
            results: Dictionary with structure:
                {
                    'MST-AV': {'MAE': {...}, 'MAPE': {...}, 'RMSE': {...}},
                    'GDRN-DFT': {...},
                    ...
                }
            conditions: List of condition columns (e.g., ['0-10', '10-25', '25-45', '45+', 'Overall'])
        """
        if conditions is None:
            conditions = ['0-10', '10-25', '25-45', '45+', 'Overall']
        
        rows = []
        for model_name, metrics in results.items():
            for metric_name, values in metrics.items():
                row = {'Model': model_name, 'Metric': metric_name}
                for condition in conditions:
                    row[condition] = values.get(condition, np.nan)
                rows.append(row)
        
        df = pd.DataFrame(rows)
        filename = f"{self.route}_performance.csv"
        
        self.s3_manager.save_results(df, 'model_performance', filename)
        print(f"Saved model performance results for {self.route}")
        
        return df
    
    def save_route_performance_detailed(self, results: Dict[str, Dict[str, Dict[str, float]]]):
        """
        Save detailed route performance (like Route 2 and Route 3 tables)
        
        Args:
            results: Dictionary with structure:
                {
                    'MST-AV': {
                        'MAE': {'0-10': 5.0, '10-25': 7.0, ..., '1-2': 5.6, ...},
                        'MAPE': {...},
                        'RMSE': {...}
                    },
                    ...
                }
        """
        # Define all columns
        normal_conditions = ['0-10', '10-25', '25-45', '45+', 'Overall']
        bus_stop_groups = ['1-2', '3-4', '5-6', '7+']
        extreme_weather = ['0-10', '10-25', '25-45', '45+', 'Overall']
        
        all_columns = normal_conditions + bus_stop_groups + extreme_weather
        
        rows = []
        for model_name, metrics in results.items():
            for metric_name, values in metrics.items():
                row = {'Model': model_name, 'Metric': metric_name}
                for col in all_columns:
                    row[col] = values.get(col, np.nan)
                rows.append(row)
        
        df = pd.DataFrame(rows)
        filename = f"{self.route}_performance.csv"
        
        self.s3_manager.save_results(df, 'model_performance', filename)
        print(f"Saved detailed performance results for {self.route}")
        
        return df
    
    def save_hybrid_comparison(self, results: Dict[str, Dict[str, Dict[str, float]]]):
        """
        Save hybrid model comparison results
        
        Args:
            results: Dictionary with structure:
                {
                    'HYB(1)': {
                        'MAE': {'Route 1': {...}, 'Route 2': {...}, 'Route 3': {...}},
                        'MAPE': {...},
                        'RMSE': {...}
                    },
                    ...
                }
        """
        rows = []
        conditions = ['0-10', '10-25', '25-45', '45+', 'Overall']
        
        for model_name, metrics in results.items():
            for metric_name, route_data in metrics.items():
                for route_name, values in route_data.items():
                    row = {'Model': model_name, 'Metric': metric_name, 'Route': route_name}
                    for condition in conditions:
                        row[condition] = values.get(condition, np.nan)
                    rows.append(row)
        
        df = pd.DataFrame(rows)
        filename = 'hybrid_model_comparison.csv'
        
        self.s3_manager.save_results(df, 'hybrid_comparison', filename)
        print("Saved hybrid model comparison results")
        
        return df
    
    def save_baseline_comparison(self, results: Dict[str, Dict[str, Dict[str, float]]]):
        """
        Save baseline model comparison results (HYB(2) vs benchmarks)
        
        Args:
            results: Dictionary with benchmark model results
        """
        rows = []
        conditions = ['0-10', '10-25', '25-45', '45+', 'Overall']
        
        for model_name, metrics in results.items():
            for metric_name, route_data in metrics.items():
                for route_name, values in route_data.items():
                    row = {'Model': model_name, 'Metric': metric_name, 'Route': route_name}
                    for condition in conditions:
                        row[condition] = values.get(condition, np.nan)
                    rows.append(row)
        
        df = pd.DataFrame(rows)
        filename = 'baseline_model_comparison.csv'
        
        self.s3_manager.save_results(df, 'baseline_comparison', filename)
        print("Saved baseline model comparison results")
        
        return df
    
    def save_ablation_study(self, results: pd.DataFrame, model_name: str):
        """
        Save ablation study results
        
        Args:
            results: DataFrame with ablation study results
            model_name: Name of the model (e.g., 'gdrn_dft', 'knn', 'fenn', 'mgcn')
        """
        filename = f"{model_name}_ablation.csv"
        
        self.s3_manager.save_results(results, 'ablation_studies', filename)
        print(f"Saved ablation study results for {model_name}")
        
        return results
    
    def save_predictions(self, predictions: pd.DataFrame, timestamp: Optional[str] = None):
        """
        Save prediction results with timestamp
        
        Args:
            predictions: DataFrame with prediction results
            timestamp: Optional timestamp string (default: current time)
        """
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        filename = f"{timestamp}_{self.route}_predictions.csv"
        
        self.s3_manager.save_results(predictions, 'predictions', filename)
        print(f"Saved predictions to {filename}")
        
        return filename
    
    def save_metrics_summary(self, metrics: Dict[str, Any], timestamp: Optional[str] = None):
        """
        Save metrics summary as JSON
        
        Args:
            metrics: Dictionary with metric values
            timestamp: Optional timestamp string
        """
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        filename = f"{timestamp}_{self.route}_metrics.json"
        
        self.s3_manager.save_metrics_json(metrics, 'predictions', filename)
        print(f"Saved metrics summary to {filename}")
        
        return filename
    
    def load_model_performance(self, route: Optional[str] = None) -> pd.DataFrame:
        """Load model performance results from S3"""
        route = route or self.route
        filename = f"{route}_performance.csv"
        return self.s3_manager.load_results('model_performance', filename)
    
    def load_hybrid_comparison(self) -> pd.DataFrame:
        """Load hybrid model comparison results from S3"""
        return self.s3_manager.load_results('hybrid_comparison', 'hybrid_model_comparison.csv')
    
    def load_baseline_comparison(self) -> pd.DataFrame:
        """Load baseline comparison results from S3"""
        return self.s3_manager.load_results('baseline_comparison', 'baseline_model_comparison.csv')
    
    def load_ablation_study(self, model_name: str) -> pd.DataFrame:
        """Load ablation study results from S3"""
        filename = f"{model_name}_ablation.csv"
        return self.s3_manager.load_results('ablation_studies', filename)
    
    def create_summary_report(self, all_results: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Create a comprehensive summary report from all results
        
        Args:
            all_results: Dictionary with all result DataFrames
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'route': self.route,
            'timestamp': datetime.now().isoformat(),
            'best_models': {},
            'overall_statistics': {}
        }
        
        # Find best models by metric
        if 'model_performance' in all_results:
            df = all_results['model_performance']
            
            for metric in ['MAE', 'MAPE', 'RMSE']:
                metric_data = df[df['Metric'] == metric]
                if not metric_data.empty and 'Overall' in metric_data.columns:
                    best_idx = metric_data['Overall'].idxmin()
                    best_model = metric_data.loc[best_idx, 'Model']
                    best_value = metric_data.loc[best_idx, 'Overall']
                    
                    summary['best_models'][metric] = {
                        'model': best_model,
                        'value': float(best_value)
                    }
        
        # Calculate overall statistics
        for result_type, df in all_results.items():
            if 'Overall' in df.columns:
                summary['overall_statistics'][result_type] = {
                    'mean': float(df['Overall'].mean()),
                    'std': float(df['Overall'].std()),
                    'min': float(df['Overall'].min()),
                    'max': float(df['Overall'].max())
                }
        
        return summary
    
    def save_summary_report(self, summary: Dict[str, Any]):
        """Save summary report to S3"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{self.route}_summary.json"
        
        self.s3_manager.save_metrics_json(summary, 'predictions', filename)
        print(f"Saved summary report to {filename}")
        
        return filename
