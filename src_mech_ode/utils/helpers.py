"""
Helper utilities for DMG tumor growth analysis.
"""
import logging
from typing import Optional
import numpy as np

def clean_predictions_array(
        predictions: np.ndarray, patient_id: str = "", logger: Optional[logging.Logger] = None,
        allow_negative: bool = True, max_value: float = 1e10
    ) -> np.ndarray:
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