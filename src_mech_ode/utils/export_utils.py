"""
Export utilities for DMG tumor growth analysis results.
"""
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

def export_bootstrap_parameters(bootstrap_results: pd.DataFrame, patient_id: str, 
                              data_type: str, mode: str, output_dir: Path, logger: logging.Logger):
    """Export bootstrap parameters to CSV with mode identifier."""
    
    if bootstrap_results.empty:
        logger.warning(f"No bootstrap results to export for {patient_id} {mode} mode")
        return None
    
    # Get only fitted model parameters (exclude metadata)
    numeric_cols = bootstrap_results.select_dtypes(include=[np.number]).columns
    param_cols = [col for col in numeric_cols 
                 if col not in ['R2_1', 'rt_time', 'noise_level'] and not col.startswith('d_')]
    
    # Extract parameters and add mode column
    params_df = bootstrap_results[param_cols].copy()
    params_df['mode'] = mode
    params_df['bootstrap_id'] = range(len(params_df))
    params_df['patient_id'] = patient_id
    params_df['data_type'] = data_type
    
    # Save individual mode file
    mode_file = output_dir / f"{patient_id}_{data_type}_bootstrap_params_{mode}.csv"
    params_df.to_csv(mode_file, index=False)
    logger.info(f"Exported {len(params_df)} bootstrap parameters for {mode} mode: {mode_file}")
    
    return params_df

def export_bootstrap_predictions(bootstrap_predictions: np.ndarray, t_plot: np.ndarray,
                               patient_id: str, data_type: str, mode: str, 
                               output_dir: Path, norm_factor: float = 1.0, logger: logging.Logger = None):
    """Export bootstrap predictions across full timeline to CSV with mode identifier."""
    
    if bootstrap_predictions.size == 0:
        if logger:
            logger.warning(f"No bootstrap predictions to export for {patient_id} {mode} mode")
        return None
    
    # bootstrap_predictions shape: (1, n_timepoints, n_bootstrap)
    if bootstrap_predictions.ndim == 3:
        predictions_array = bootstrap_predictions[0, :, :]  # (n_timepoints, n_bootstrap)
    else:
        predictions_array = bootstrap_predictions
    
    n_timepoints, n_bootstrap = predictions_array.shape
    
    # Create long-format DataFrame
    data_rows = []
    for bootstrap_id in range(n_bootstrap):
        for time_idx, time_val in enumerate(t_plot):
            # Denormalize prediction
            pred_value = predictions_array[time_idx, bootstrap_id] * norm_factor
            
            data_rows.append({
                'time': time_val,
                'prediction_value': pred_value,
                'bootstrap_id': bootstrap_id,
                'mode': mode,
                'patient_id': patient_id,
                'data_type': data_type
            })
    
    predictions_df = pd.DataFrame(data_rows)
    
    # Save individual mode file
    mode_file = output_dir / f"{patient_id}_{data_type}_bootstrap_predictions_{mode}.csv"
    predictions_df.to_csv(mode_file, index=False)
    if logger:
        logger.info(f"Exported {len(predictions_df)} bootstrap predictions for {mode} mode: {mode_file}")
    
    return predictions_df

def merge_bootstrap_exports(all_params_df: pd.DataFrame, train_params_df: pd.DataFrame,
                          all_preds_df: pd.DataFrame, train_preds_df: pd.DataFrame,
                          patient_id: str, data_type: str, output_dir: Path, logger: logging.Logger):
    """Merge bootstrap parameters and predictions from both modes."""
    
    # Merge parameters
    if all_params_df is not None and train_params_df is not None:
        unified_params = pd.concat([all_params_df, train_params_df], ignore_index=True)
        unified_params_file = output_dir / f"{patient_id}_{data_type}_bootstrap_params_unified.csv"
        unified_params.to_csv(unified_params_file, index=False)
        logger.info(f"Created unified parameters file: {unified_params_file}")
    
    # Merge predictions
    if all_preds_df is not None and train_preds_df is not None:
        unified_preds = pd.concat([all_preds_df, train_preds_df], ignore_index=True)
        unified_preds_file = output_dir / f"{patient_id}_{data_type}_bootstrap_predictions_unified.csv"
        unified_preds.to_csv(unified_preds_file, index=False)
        logger.info(f"Created unified predictions file: {unified_preds_file}")