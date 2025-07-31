"""
Plotting utilities for DMG tumor growth analysis.
"""
import logging
from pathlib import Path
from typing import Dict
import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error, r2_score

from .config import P_PERCENTILE
from .helpers import clean_predictions_array

# Configure matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'

def plot_model_fit_with_uncertainty(
        data_dict: Dict, bootstrap_predictions: np.ndarray, 
        patient_id: str, output_dir: Path, data_type: str, mode: str, logger: logging.Logger
    ):
    """Create publication-quality plot showing model fit with uncertainty bands."""
    
    plt.figure(figsize=(10, 8))
    
    # Get denormalized data for plotting
    time_data = data_dict['v_1'].index.values
    volume_data = data_dict['v_1'].values * data_dict['norm_factor']
    error_data = data_dict['dv_1'].values * data_dict['norm_factor']
    rt_time = data_dict['rt_time']
    rt_end = data_dict.get('rt_end', rt_time + 42)
    
    # Plot experimental data
    if mode == 'prediction':
        plt.errorbar(
            time_data, volume_data, 
            fmt='o', color='blue', markersize=8, capsize=5, 
            label='Experimental Data'
        )
    else:
        plt.errorbar(
            time_data, volume_data, yerr=error_data,
            fmt='o', color='blue', markersize=8, capsize=5, 
            label='Experimental Data'
        )
    
    # Plot uncertainty bands from bootstrap
    if bootstrap_predictions.size > 0:
        t_min, t_max = time_data.min(), time_data.max()
        t_plot = np.linspace(t_min, t_max, bootstrap_predictions.shape[1])
        
        predictions_for_well = bootstrap_predictions[0, :, :]
        predictions_denorm = predictions_for_well * data_dict['norm_factor']
        
        median_volume = np.nanmedian(predictions_denorm, axis=1)
        percentile_975 = np.nanpercentile(predictions_denorm, 97.5, axis=1)
        percentile_025 = np.nanpercentile(predictions_denorm, 2.5, axis=1)
        
        # Plot uncertainty band
        plt.fill_between(t_plot, percentile_025, percentile_975, 
                        alpha=0.25, color='red', linewidth=0, label='95% CI')
        plt.plot(t_plot, median_volume, color='red', linewidth=2, 
                linestyle='-', label='Model Median')
        
        # Calculate R² at data points
        median_at_data_times = np.interp(time_data, t_plot, median_volume)
        r2 = r2_score(volume_data, median_at_data_times)
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
                fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add RT timing markers
    plt.axvline(x=rt_time, color='gray', linestyle='--', alpha=0.7, linewidth=2, 
               label=f'RT start (day {rt_time:.0f})')
    plt.axvline(x=rt_end, color='orange', linestyle='--', alpha=0.7, linewidth=2, 
               label=f'RT end (day {rt_end:.0f})')
    
    plt.xlabel('Time (days)', fontsize=14)
    plt.ylabel('Volume (mm³)', fontsize=14)
    plt.title(f'Tumor Growth Model Fit - {patient_id} ({data_type}) - {mode.title()} Mode', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_file = output_dir / f"{patient_id}_{data_type}_model_fit.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved model fit plot: {plot_file}")

def plot_parameter_distributions(bootstrap_results: pd.DataFrame, patient_id: str, 
                               output_dir: Path, data_type: str, logger: logging.Logger):
    """Plot parameter distributions from bootstrap analysis."""
    
    # Get parameter columns (exclude metadata)
    numeric_cols = bootstrap_results.select_dtypes(include=[np.number]).columns
    param_cols = [col for col in numeric_cols 
                 if col not in ['R2_1', 'rt_time', 'noise_level'] and not col.startswith('d_')]
    
    if len(param_cols) == 0:
        logger.warning("No parameters found for distribution plotting")
        return
    
    n_params = len(param_cols)
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_params == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, param in enumerate(param_cols):
        ax = axes[i]
        values = bootstrap_results[param].dropna()
        
        if len(values) > 5:
            # Check for sufficient variance for KDE
            if values.std() > 1e-10 and len(values.unique()) > 3:
                try:
                    # Create smooth distribution plot
                    kde = gaussian_kde(values)
                    x_range = np.linspace(values.min(), values.max(), 200)
                    density = kde(x_range)
                    
                    ax.fill_between(x_range, density, alpha=0.6, color='skyblue')
                    ax.plot(x_range, density, color='navy', linewidth=2)
                    
                    # Add statistics
                    mean_val = values.mean()
                    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                              label=f'Mean: {mean_val:.3f}')
                    
                    ci_lower, ci_upper = np.percentile(values, [2.5, 97.5])
                    ax.axvspan(ci_lower, ci_upper, alpha=0.2, color='gray', label='95% CI')
                    
                except:
                    # Fallback to histogram
                    ax.hist(values, bins=min(20, len(values)//2), alpha=0.6, color='skyblue')
                    ax.axvline(values.mean(), color='red', linestyle='--', linewidth=2)
            else:
                # Use histogram for low-variance data
                ax.hist(values, bins=min(10, len(values)//2), alpha=0.6, color='skyblue')
                ax.axvline(values.mean(), color='red', linestyle='--', linewidth=2)
        else:
            # Too few points for meaningful distribution
            ax.hist(values, bins=min(5, len(values)), alpha=0.6, color='skyblue')
        
        ax.set_xlabel(param, fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{param} Distribution', fontsize=12)
        
        # Only add legend if there are labeled artists
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'{patient_id} - Parameter Distributions (n={len(bootstrap_results)})', 
                 fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save plot
    plot_file = output_dir / f"{patient_id}_{data_type}_parameter_distributions.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved parameter distributions plot: {plot_file}")

def plot_estimation_prediction_distribution(
        train_data: Dict, test_data: Dict, all_test_predictions: Dict, 
        patient_id: str, output_dir: Path, data_type: str, logger: logging.Logger
    ):
    """Plot KDE distribution of predictions with ground truth vertical line."""
    
    # Import here to avoid circular imports
    from .predict import get_prediction_t
    
    plt.figure(figsize=(10, 6))
    
    rt_time = train_data['rt_time']
    norm_factor = train_data['norm_factor']
    
    # Get test data (already denormalized)
    test_times = test_data['v_1'].index.values
    test_volumes = test_data['v_1'].values * norm_factor
    
    # Extract predictions at test times - ensure it's a flat array
    try:
        all_predictions = get_prediction_t(all_test_predictions, test_times, logger)
        # If it's still nested, flatten it
        if isinstance(all_predictions, (list, tuple)) and len(all_predictions) > 0:
            if hasattr(all_predictions[0], '__len__') and not isinstance(all_predictions[0], (str, bytes)):
                all_predictions = np.concatenate(all_predictions)
        all_predictions = np.asarray(all_predictions).flatten()

        all_predictions = clean_predictions_array(all_predictions, patient_id, logger)
        
        # NOTE: all_test_predictions from generate_predictions_multi_step are ALREADY denormalized
        # (they were multiplied by norm_factor in generate_predictions_multi_step)
        # So we DON'T multiply by norm_factor again here
        
    except Exception as e:
        logger.warning(f"Failed to extract predictions: {e}")
        all_predictions = np.array([])

    if len(all_predictions) > 0:
        # Validate predictions are in reasonable range
        if np.any(all_predictions < 0):
            logger.warning(f"Found negative predictions, clipping to 0")
            all_predictions = np.maximum(all_predictions, 0)
        
        # Create KDE plot
        if len(all_predictions) > 5 and np.std(all_predictions) > 1e-10:
            try:
                kde = gaussian_kde(all_predictions)
                x_range = np.linspace(all_predictions.min(), all_predictions.max(), 200)
                density = kde(x_range)
                
                plt.fill_between(x_range, density, alpha=0.6, color='lightblue')
                plt.plot(x_range, density, color='blue', linewidth=2)
            except Exception as e:
                logger.warning(f"KDE failed, using histogram: {e}")
                plt.hist(all_predictions, bins=min(20, len(all_predictions)//2), alpha=0.6, 
                        color='lightblue', density=True)
        else:
            plt.hist(all_predictions, bins=min(20, max(5, len(all_predictions)//2)), alpha=0.6, 
                    color='lightblue', density=True)
        
        # Add ground truth vertical line
        for i, test_vol in enumerate(test_volumes):
            plt.axvline(test_vol, color='red', linestyle='-', linewidth=3, 
                       label=f'Ground Truth {i+1}' if len(test_volumes) > 1 else 'Ground Truth')
        
        # Add statistics
        median_pred = np.median(all_predictions)
        plt.axvline(median_pred, color='green', linestyle='--', linewidth=2, label='Median Prediction')
        
        # Add statistics text
        try:
            rmse = np.sqrt(mean_squared_error(test_volumes, [median_pred] * len(test_volumes)))
            stats_text = f'Median: {median_pred:.1f}\nRMSE: {rmse:.3f}\nN samples: {len(all_predictions)}'
            plt.text(0.7, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), verticalalignment='top')
        except Exception as e:
            logger.warning(f"Failed to calculate RMSE: {e}")
            stats_text = f'Median: {median_pred:.1f}\nN samples: {len(all_predictions)}'
            plt.text(0.7, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), verticalalignment='top')
    else:
        plt.text(0.5, 0.5, 'No predictions available', transform=plt.gca().transAxes, 
                fontsize=14, ha='center', va='center')
    
    plt.xlabel('Volume (mm³)', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.title(f'{patient_id} - Test Prediction Distribution ({data_type})', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_file = output_dir / f"{patient_id}_{data_type}_estimation_distribution.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved estimation distribution plot: {plot_file}")

def plot_estimation_train_test(train_data: Dict, test_data: Dict, cont_predictions: Dict, 
                               patient_id: str, output_dir: Path, data_type: str, logger: logging.Logger):
    """Plot estimation mode with uncertainty bands like prediction mode, plus X for predicted test point."""
    
    plt.figure(figsize=(12, 8))
    
    rt_time = train_data['rt_time']
    rt_end = train_data.get('rt_end', rt_time + 42)
    norm_factor = train_data['norm_factor']
    
    # Plot training data (dots with error bars)
    train_times = train_data['v_1'].index.values
    train_volumes = train_data['v_1'].values * norm_factor
    train_errors = train_data['dv_1'].values * norm_factor
    
    plt.errorbar(train_times, train_volumes, yerr=train_errors, 
                fmt='o', color='blue', markersize=8, capsize=5, label='Training Data')
    
    # Plot test data (different colored dot with error bars)
    test_times = test_data['v_1'].index.values
    test_volumes = test_data['v_1'].values * norm_factor
    test_errors = test_data['dv_1'].values * norm_factor
    
    plt.errorbar(test_times, test_volumes, yerr=test_errors, 
                fmt='o', color='green', markersize=10, capsize=5, label='Test Data (Actual)')
    
    all_continuous_predictions = cont_predictions["y_sol"].copy()
    t_plot = cont_predictions["y_time"].copy()
            
    # Calculate continuous uncertainty bands
    median_volumes = np.nanmedian(all_continuous_predictions, axis=0)
    percentile_high = np.nanpercentile(all_continuous_predictions, 100 - P_PERCENTILE, axis=0)
    percentile_low = np.nanpercentile(all_continuous_predictions, P_PERCENTILE, axis=0)
    
    # Plot continuous uncertainty band
    plt.fill_between(t_plot, percentile_low, percentile_high, 
                    alpha=0.25, color='red', linewidth=0, label=f'{100 - 2*P_PERCENTILE}% CI')
    plt.plot(t_plot, median_volumes, color='red', linewidth=2, 
            linestyle='-', label='Model Median')
    
    # Calculate R² for training data
    median_at_train_times = np.interp(train_times, t_plot, median_volumes)
    train_r2 = r2_score(train_volumes, median_at_train_times)
    plt.text(0.05, 0.95, f'Train R² = {train_r2:.3f}', transform=plt.gca().transAxes, 
            fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Get median prediction at test timepoint (from continuous curve)
    median_at_test_times = np.interp(test_times, t_plot, median_volumes)
    
    # Plot X marker at test timepoint (sits exactly on continuous median curve)
    plt.scatter(test_times, median_at_test_times, 
                marker='x', color='darkred', s=200, linewidth=4, 
                label='Test Prediction (Median)', zorder=10)
    
    # Calculate RMSE using continuous median prediction
    rmse = np.sqrt(mean_squared_error(test_volumes, median_at_test_times))
    plt.text(0.05, 0.88, f'Test RMSE = {rmse:.3f}', transform=plt.gca().transAxes, 
            fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add RT timing markers
    plt.axvline(x=rt_time, color='gray', linestyle='--', alpha=0.7, linewidth=2, 
               label=f'RT start (day {rt_time:.0f})')
    plt.axvline(x=rt_end, color='orange', linestyle='--', alpha=0.7, linewidth=2, 
               label=f'RT end (day {rt_end:.0f})')
    
    plt.xlabel('Time (days)', fontsize=14)
    plt.ylabel('Volume (mm³)', fontsize=14)
    plt.title(f'{patient_id} - Estimation Mode: Train/Test Analysis ({data_type})', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_file = output_dir / f"{patient_id}_{data_type}_estimation_train_test.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved estimation train/test plot: {plot_file}")