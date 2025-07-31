"""
Prediction generation methods for DMG tumor growth analysis.
"""
import logging
from typing import Dict, List, Tuple
import numpy as np

from .model import solve_tumor_dynamics

def generate_predictions_single_step(fitted_params_list: List[Dict], test_times: np.ndarray, 
                                    rt_time: float, norm_factor: float, no_volume: bool) -> np.ndarray:
    """Generate test predictions using single-step method (direct calculation at test points)."""
    all_predictions = []
    for fitted_params in fitted_params_list:
        try:
            test_model_solution = solve_tumor_dynamics(test_times, fitted_params, rt_time, fitted_params['SD_1'], no_volume)
            test_model_volumes = test_model_solution.sum(axis=1) * norm_factor
            all_predictions.extend(test_model_volumes)
        except:
            continue
    return np.array(all_predictions)

def get_prediction_t(y_solution: Dict, test_times: np.ndarray, logger: logging.Logger = None) -> np.ndarray:
    """
    Get the prediction at test_times via interpolation.
    
    Args:
        y_solution: Dict with keys 'y_sol' and 'y_time'
                   y_sol can be:
                   - 1D array: (len(y_time),) - single solution
                   - 2D array: (n_bootstrap, len(y_time)) - multiple bootstrap solutions
        test_times: Array of times where we want predictions
        logger: Optional logger for error reporting
        
    Returns:
        np.ndarray: Interpolated predictions
                   - If y_sol is 1D: returns (len(test_times),)
                   - If y_sol is 2D: returns (n_bootstrap * len(test_times),) - flattened
    """
    try:
        y_sol = y_solution["y_sol"]
        y_time = y_solution["y_time"]
        
        # Convert to numpy arrays if needed
        y_sol = np.asarray(y_sol)
        y_time = np.asarray(y_time)
        test_times = np.asarray(test_times)
        
        # Handle different dimensionalities
        if y_sol.ndim == 1:
            # Single solution case: (len(y_time),)
            y_test = np.interp(test_times, y_time, y_sol)
            return y_test
            
        elif y_sol.ndim == 2:
            # Multiple bootstrap solutions case: (n_bootstrap, len(y_time))
            n_bootstrap, n_timepoints = y_sol.shape
            
            # Validate dimensions
            if len(y_time) != n_timepoints:
                raise ValueError(f"y_time length {len(y_time)} doesn't match y_sol timepoints {n_timepoints}")
            
            # Interpolate each bootstrap solution
            all_interpolated = []
            for i in range(n_bootstrap):
                y_interp = np.interp(test_times, y_time, y_sol[i, :])
                all_interpolated.extend(y_interp)  # Flatten by extending
            
            return np.array(all_interpolated)
            
        else:
            raise ValueError(f"y_sol has unsupported dimensionality: {y_sol.ndim}D. Expected 1D or 2D.")
            
    except Exception as e:
        if logger:
            logger.warning(f"Error in get_prediction_t: {e}")
            logger.warning(f"y_sol shape: {y_solution.get('y_sol', 'N/A')}")
            logger.warning(f"y_time shape: {y_solution.get('y_time', 'N/A')}")
        return np.array([])

def generate_predictions_multi_step(fitted_params_list: List[Dict], train_times: np.ndarray, 
                                   test_times: np.ndarray, rt_time: float, norm_factor: float,
                                   no_volume: bool) -> Dict:
    """Generate test predictions using multi-step method (continuous timeline with interpolation)."""
    all_times = np.concatenate([train_times, test_times])
    t_min, t_max = all_times.min(), all_times.max()
    t_max = t_max + 20 # tmax + 20 and then trim at tmax
    t_plot = np.linspace(t_min, t_max, int(t_max - t_min) + 1)

    all_predictions = []
    for fitted_params in fitted_params_list:
        try:
            # Generate continuous prediction
            continuous_solution = solve_tumor_dynamics(t_plot, fitted_params, rt_time, fitted_params['SD_1'], no_volume)
            # sum over all states
            continuous_volumes = continuous_solution.sum(axis=1) * norm_factor
            all_predictions.append(continuous_volumes)
        except:
            continue
    
    return { 
        "y_sol": np.array(all_predictions),
        "y_time": t_plot
    }

def get_median_solution(y: Dict, test_t: np.ndarray, logger: logging.Logger = None) -> Tuple[Dict, float]:
    """
    Get median solution and median prediction at test times.
    
    Args:
        y: Dict with 'y_sol' (n_bootstrap, len(y_time)) and 'y_time'
        test_t: Test times where we want the median prediction
        logger: Optional logger for error reporting
        
    Returns:
        Tuple of:
        - Dict with median solution over full timeline
        - Float: median prediction at test times (scalar if single test point)
    """
    try:
        y_sol = y["y_sol"]  # (n_bootstrap, len(y_time))
        y_time = y["y_time"]
        
        # Convert to numpy arrays
        y_sol = np.asarray(y_sol)
        y_time = np.asarray(y_time)
        test_t = np.asarray(test_t)
        
        if y_sol.ndim != 2:
            raise ValueError(f"Expected 2D y_sol, got {y_sol.ndim}D")
        
        # Calculate median across bootstrap samples (axis=0)
        y_median = np.median(y_sol, axis=0)  # Shape: (len(y_time),)
        
        # Get median prediction at test times
        median_prediction_at_test = np.interp(test_t, y_time, y_median)
        
        # If single test point, return scalar
        if len(test_t) == 1:
            median_prediction_scalar = float(median_prediction_at_test[0])
        else:
            # Multiple test points - return first one as representative
            median_prediction_scalar = float(median_prediction_at_test[0])
        
        # Return dict with median solution for plotting
        y_median_sol = {"y_sol": y_median, "y_time": y_time}
        
        return y_median_sol, median_prediction_scalar
        
    except Exception as e:
        if logger:
            logger.warning(f"Error in get_median_solution: {e}")
            logger.warning(f"y_sol shape: {y.get('y_sol', 'N/A')}")
        return {"y_sol": np.array([]), "y_time": np.array([])}, 0.0