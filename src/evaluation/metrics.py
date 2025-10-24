"""
Evaluation Metrics Module
Implements MAE, MAPE, and RMSE calculations for ETA predictions
"""

import numpy as np
from typing import Union, Tuple


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error (MAE)
    
    MAE = (1/n) * Σ|y_true - y_pred|
    
    Args:
        y_true: True ETA values (in minutes)
        y_pred: Predicted ETA values (in minutes)
    
    Returns:
        MAE value
    """
    return np.mean(np.abs(y_true - y_pred))


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray, 
                                   epsilon: float = 1e-10) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE)
    
    MAPE = (100/n) * Σ|(y_true - y_pred) / y_true|
    
    Args:
        y_true: True ETA values (in minutes)
        y_pred: Predicted ETA values (in minutes)
        epsilon: Small value to avoid division by zero
    
    Returns:
        MAPE value (as percentage)
    """
    # Avoid division by zero
    y_true_safe = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error (RMSE)
    
    RMSE = sqrt((1/n) * Σ(y_true - y_pred)²)
    
    Args:
        y_true: True ETA values (in minutes)
        y_pred: Predicted ETA values (in minutes)
    
    Returns:
        RMSE value
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate all evaluation metrics
    
    Args:
        y_true: True ETA values (in minutes)
        y_pred: Predicted ETA values (in minutes)
    
    Returns:
        Dictionary with MAE, MAPE, and RMSE
    """
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'RMSE': root_mean_squared_error(y_true, y_pred)
    }


def calculate_metrics_by_condition(y_true: np.ndarray, y_pred: np.ndarray, 
                                   conditions: np.ndarray) -> dict:
    """
    Calculate metrics grouped by conditions (e.g., ETA ranges)
    
    Args:
        y_true: True ETA values (in minutes)
        y_pred: Predicted ETA values (in minutes)
        conditions: Array of condition labels for each prediction
    
    Returns:
        Dictionary with metrics for each condition
    """
    results = {}
    
    for condition in np.unique(conditions):
        mask = conditions == condition
        if np.sum(mask) > 0:
            results[condition] = calculate_all_metrics(
                y_true[mask], 
                y_pred[mask]
            )
    
    # Calculate overall metrics
    results['Overall'] = calculate_all_metrics(y_true, y_pred)
    
    return results


def categorize_eta_ranges(eta_values: np.ndarray) -> np.ndarray:
    """
    Categorize ETA values into ranges as per the paper
    
    Ranges:
    - 0-10 minutes
    - 10-25 minutes
    - 25-45 minutes
    - 45+ minutes
    
    Args:
        eta_values: Array of ETA values in minutes
    
    Returns:
        Array of category labels
    """
    categories = np.empty(len(eta_values), dtype=object)
    
    categories[eta_values < 10] = '0-10'
    categories[(eta_values >= 10) & (eta_values < 25)] = '10-25'
    categories[(eta_values >= 25) & (eta_values < 45)] = '25-45'
    categories[eta_values >= 45] = '45+'
    
    return categories


def categorize_bus_stop_groups(stop_indices: np.ndarray, total_stops: int) -> np.ndarray:
    """
    Categorize bus stops into groups
    
    Groups:
    - 1-2: First quarter
    - 3-4: Second quarter
    - 5-6: Third quarter
    - 7+: Fourth quarter
    
    Args:
        stop_indices: Array of bus stop indices (0-based)
        total_stops: Total number of stops on the route
    
    Returns:
        Array of group labels
    """
    categories = np.empty(len(stop_indices), dtype=object)
    
    quarter = total_stops / 4
    
    categories[stop_indices < quarter] = '1-2'
    categories[(stop_indices >= quarter) & (stop_indices < 2 * quarter)] = '3-4'
    categories[(stop_indices >= 2 * quarter) & (stop_indices < 3 * quarter)] = '5-6'
    categories[stop_indices >= 3 * quarter] = '7+'
    
    return categories


def evaluate_model_performance(y_true: np.ndarray, y_pred: np.ndarray, 
                               eta_ranges: bool = True) -> dict:
    """
    Comprehensive model evaluation with ETA range categorization
    
    Args:
        y_true: True ETA values (in minutes)
        y_pred: Predicted ETA values (in minutes)
        eta_ranges: Whether to categorize by ETA ranges
    
    Returns:
        Dictionary with comprehensive metrics
    """
    results = {
        'overall': calculate_all_metrics(y_true, y_pred)
    }
    
    if eta_ranges:
        conditions = categorize_eta_ranges(y_true)
        results['by_eta_range'] = calculate_metrics_by_condition(
            y_true, y_pred, conditions
        )
    
    return results


def compare_models(model_results: dict) -> dict:
    """
    Compare multiple models and rank them
    
    Args:
        model_results: Dictionary with model names as keys and their metrics as values
            Example: {'MST-AV': {'MAE': 4.3, 'MAPE': 13.2, ...}, ...}
    
    Returns:
        Dictionary with rankings and comparisons
    """
    comparison = {
        'rankings': {},
        'best_model': {},
        'improvements': {}
    }
    
    # Rank models by each metric
    for metric in ['MAE', 'MAPE', 'RMSE']:
        metric_values = {
            model: results.get(metric, float('inf'))
            for model, results in model_results.items()
        }
        
        # Sort by metric value (lower is better)
        ranked = sorted(metric_values.items(), key=lambda x: x[1])
        comparison['rankings'][metric] = ranked
        comparison['best_model'][metric] = ranked[0][0]
        
        # Calculate improvements over baseline (first model)
        if len(ranked) > 1:
            baseline_value = ranked[-1][1]  # Worst model
            improvements = {}
            for model, value in ranked[:-1]:
                improvement = ((baseline_value - value) / baseline_value) * 100
                improvements[model] = improvement
            comparison['improvements'][metric] = improvements
    
    return comparison
