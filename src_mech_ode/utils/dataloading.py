"""
Data loading and preprocessing utilities for DMG tumor growth analysis.
"""
import pickle
import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from .config import PATIENT_CONFIG

def load_data(data_path: str, logger: logging.Logger) -> Dict:
    """Load tumor data from pickle file."""
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    logger.info(f"Loaded data for {len(data)} patients: {list(data.keys())}")
    return data

def detect_available_slices(patient_df: pd.DataFrame) -> Dict[str, List[str]]:
    """Detect available slice types for a patient and return structured info."""
    slice_info = {
        'volume': [],
        'initial_slices': [],
        'largest_slices': []
    }
    
    # Check for volume data
    if 'volume_manual' in patient_df.columns:
        slice_info['volume'].append('volume_manual')
    
    # Detect initial slice columns
    initial_cols = [col for col in patient_df.columns if col.startswith('slice_initial_')]
    slice_info['initial_slices'] = sorted([col.replace('slice_initial_', '') for col in initial_cols])
    
    # Detect largest slice columns
    largest_cols = [col for col in patient_df.columns if col.startswith('slice_largest_')]
    slice_info['largest_slices'] = sorted([col.replace('slice_largest_', '') for col in largest_cols])
    
    return slice_info

def print_patient_slice_info(patient_id: str, patient_df: pd.DataFrame):
    """Print available slices for a patient."""
    slice_info = detect_available_slices(patient_df)
    
    print(f"\n📊 Available data for {patient_id}:")
    print(f"  Volume data: {len(slice_info['volume'])} columns")
    if slice_info['volume']:
        print(f"    - {slice_info['volume']}")
    
    print(f"  Initial slices: {len(slice_info['initial_slices'])} slices")
    if slice_info['initial_slices']:
        print(f"    - Slice IDs: {slice_info['initial_slices']}")
    
    print(f"  Largest slices: {len(slice_info['largest_slices'])} slices")
    if slice_info['largest_slices']:
        print(f"    - Slice IDs: {slice_info['largest_slices']}")

def calculate_rt_timeline(patient_df: pd.DataFrame) -> Dict:
    """Calculate RT timeline from patient data."""
    first_imaging_day = patient_df['time(rel_days)'].iloc[0]
    days_until_rt_ends_at_first_imaging = patient_df['time_end_RT(rel_days)'].iloc[0]
    
    if days_until_rt_ends_at_first_imaging < 0:
        days_from_first_imaging_to_rt_end = abs(days_until_rt_ends_at_first_imaging)
        rt_end_day = first_imaging_day + days_from_first_imaging_to_rt_end
        timeline_type = "prospective"
    else:
        days_since_rt_ended_at_first_imaging = days_until_rt_ends_at_first_imaging
        rt_end_day = first_imaging_day - days_since_rt_ended_at_first_imaging
        timeline_type = "retrospective"
    
    rt_start_day = rt_end_day - 42  # Standard 6-week RT course
    
    return {
        'rt_start': rt_start_day,
        'rt_end': rt_end_day,
        'timeline_type': timeline_type,
        'explanation': f"RT: Day {rt_start_day:.0f} to Day {rt_end_day:.0f} [{timeline_type}]"
    }

def prepare_patient_data(patient_df: pd.DataFrame, data_type: str = "volume", 
                        time_column: str = "time(rel_days)", 
                        normalization: bool = True, noise_level: float = 0.01) -> Dict:
    """Prepare patient data for analysis."""
    
    # Determine target column based on data_type
    if data_type == 'volume':
        target_column = 'volume_manual'
    elif data_type.startswith('initial_'):
        slice_id = data_type.replace('initial_', '')
        target_column = f'slice_initial_{slice_id}'
    elif data_type.startswith('largest_'):
        slice_id = data_type.replace('largest_', '')
        target_column = f'slice_largest_{slice_id}'
    else:
        raise ValueError(f"Unknown data_type: {data_type}")
    
    if target_column not in patient_df.columns:
        available_cols = [col for col in patient_df.columns if col.startswith(('volume_', 'slice_'))]
        raise ValueError(f"Column '{target_column}' not found. Available: {available_cols}")
    
    # Get RT timeline
    rt_timeline = calculate_rt_timeline(patient_df)
    
    # Extract and normalize data
    time_series = patient_df.set_index(time_column)[target_column].dropna()
    
    if normalization:
        norm_factor = time_series.iloc[0]
        normalized_data = time_series / norm_factor
    else:
        norm_factor = 1.0
        normalized_data = time_series
    
    # all target columns are mapped to the v_1, dv_1 and handled thereafter 
    return {
        "v_1": normalized_data,
        "dv_1": normalized_data * noise_level,
        "rt_time": rt_timeline['rt_start'],
        "rt_start": rt_timeline['rt_start'],
        "rt_end": rt_timeline['rt_end'],
        "norm_factor": norm_factor,
        "timeline_info": rt_timeline
    }

def apply_train_test_split(data_dict: Dict, patient_id: str, mode: str = 'prediction') -> Tuple[Dict, Optional[Dict]]:
    """Apply train/test split based on patient configuration."""
    
    if patient_id not in PATIENT_CONFIG or not PATIENT_CONFIG[patient_id]['valid']:
        return None, None
    
    config = PATIENT_CONFIG[patient_id]
    
    # Identify all volume/slice data columns (v_* and dv_*)
    volume_cols = [key for key in data_dict.keys() if key.startswith('v_') or key.startswith('dv_')]
    non_volume_cols = [key for key in data_dict.keys() if not key.startswith('v_') and not key.startswith('dv_')]
    
    # First, remove 'remove_last' points completely from the dataset if specified
    processed_data = {}
    
    # Copy non-volume data as-is
    for key in non_volume_cols:
        processed_data[key] = data_dict[key]
    
    # Process all volume/slice data
    for key in volume_cols:
        volume_data = data_dict[key].copy()
        
        if 'remove_last' in config and config['remove_last'] > 0:
            remove_last = config['remove_last']
            if len(volume_data) > remove_last:
                volume_data = volume_data.iloc[:-remove_last]
        
        processed_data[key] = volume_data
    
    if mode == 'prediction':
        # Use all remaining data as training data
        return processed_data, None
    
    ###############
    # Estimation mode - apply train/test split to remaining data
    ###############
    train_data = {}
    test_data = {}
    
    # Copy non-volume data
    for key in non_volume_cols:
        train_data[key] = processed_data[key]
        test_data[key] = processed_data[key]
    
    # Handle volume/slice data based on configuration
    for key in volume_cols:
        volume_data = processed_data[key]
        
        if config['mode'] == 'last_n_points':
            n_points = config['n_points']
            if len(volume_data) > n_points:
                train_data[key] = volume_data.iloc[:-n_points]
                test_data[key] = volume_data.iloc[-n_points:]
            else:
                train_data[key] = volume_data.iloc[:1]
                test_data[key] = volume_data.iloc[-1:]
        
        elif config['mode'] == 'remove_last_predict':
            predict_remaining = config['predict_remaining']
            
            if len(volume_data) > predict_remaining:
                train_end = -predict_remaining
                train_data[key] = volume_data.iloc[:train_end]
                test_data[key] = volume_data.iloc[train_end:]
            else:
                train_data[key] = volume_data.iloc[:1]
                test_data[key] = volume_data.iloc[1:] if len(volume_data) > 1 else volume_data.iloc[:1]
        
        elif config['mode'] == 'pre_rt_points':
            rt_time = processed_data['rt_time']
            train_mask = volume_data.index <= rt_time
            test_mask = volume_data.index > rt_time
            
            train_data[key] = volume_data[train_mask]
            test_data[key] = volume_data[test_mask]
        
        else:
            # Default: use all remaining data for training
            train_data[key] = volume_data
            test_data[key] = volume_data.iloc[-1:]
    
    return train_data, test_data

def clean_predictions_array(predictions: np.ndarray, patient_id: str = "", logger: Optional[logging.Logger] = None,
                          allow_negative: bool = True, max_value: float = 1e10) -> np.ndarray:
    """
    Clean predictions array by removing NaN, inf, and optionally negative values.
    Preserves original array shape and structure.
    
    Args:
        predictions: Input array to clean (any shape)
        patient_id: Patient ID for logging
        logger: Logger instance for output
        allow_negative: Whether to keep negative values
        max_value: Maximum allowed value (clip above this)
        
    Returns:
        Cleaned numpy array with same shape, problematic values replaced with 0
    """
    try:
        # Convert to numpy array but preserve shape
        predictions = np.asarray(predictions)
        original_shape = predictions.shape
        
        # Log initial state
        if logger:
            logger.info(f"{patient_id} - Raw predictions shape: {original_shape}")
            if predictions.size > 0:
                logger.info(f"{patient_id} - Raw predictions range: [{np.min(predictions):.3e}, {np.max(predictions):.3e}]")
        
        # Check for problematic values
        nan_mask = np.isnan(predictions)
        inf_mask = np.isinf(predictions)
        neg_mask = predictions < 0
        large_mask = np.abs(predictions) > max_value
        
        # Count and log issues
        n_nan = np.sum(nan_mask)
        n_inf = np.sum(inf_mask)
        n_neg = np.sum(neg_mask)
        n_large = np.sum(large_mask)
        
        if logger:
            if n_nan > 0:
                logger.warning(f"{patient_id} - Found {n_nan} NaN values, replacing with 0")
            if n_inf > 0:
                logger.warning(f"{patient_id} - Found {n_inf} infinite values, replacing with 0")
            if n_neg > 0 and not allow_negative:
                logger.warning(f"{patient_id} - Found {n_neg} negative values, replacing with 0")
            if n_large > 0:
                logger.warning(f"{patient_id} - Found {n_large} extremely large values, clipping to ±{max_value:.1e}")
        
        # Clean the array in place
        clean_predictions = predictions.copy()
        
        # Replace problematic values with 0
        clean_predictions[nan_mask] = 0
        clean_predictions[inf_mask] = 0
        if not allow_negative:
            clean_predictions[neg_mask] = 0
        
        # Clip large values
        clean_predictions = np.clip(clean_predictions, -max_value, max_value)
        
        # Log final state
        if logger and clean_predictions.size > 0:
            logger.info(f"{patient_id} - Clean predictions range: [{np.min(clean_predictions):.3e}, {np.max(clean_predictions):.3e}]")
        
        return clean_predictions
        
    except Exception as e:
        if logger:
            logger.error(f"{patient_id} - Error cleaning predictions: {e}")
        return np.zeros_like(predictions) if hasattr(predictions, 'shape') else np.array([])