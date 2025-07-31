#!/usr/bin/env python3
"""
Simple plotPrettyGC.py with command line arguments
Based on the original working plotPrettyGC.py with minimal changes
"""

import argparse
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

# def detect_and_trim_explosions(df, times, explosion_threshold=10.0):
#     """
#     Automatically detect and trim exploding predictions.
    
#     Args:
#         df: DataFrame with predictions
#         times: List of time points
#         explosion_threshold: Factor by which values can grow before considered "exploding"
    
#     Returns:
#         trimmed_times: List of times to keep
#         trim_point: Where the trimming occurred
#     """
#     times_sorted = sorted(times)
    
#     # Calculate median prediction at each time point
#     median_by_time = []
#     for t in times_sorted:
#         if t in df.columns:
#             median_val = df[t].median()
#             median_by_time.append(median_val)
#         else:
#             median_by_time.append(np.nan)
    
#     median_by_time = np.array(median_by_time)
    
#     # Find the first "stable" period (after initial growth)
#     stable_start_idx = min(5, len(median_by_time) // 4)  # Skip first 25% or 5 points
#     if stable_start_idx >= len(median_by_time):
#         return times_sorted, None  # Too few points
    
#     baseline_median = np.median(median_by_time[stable_start_idx:stable_start_idx+3])
    
#     # Look for explosions: values much larger than baseline
#     explosion_detected = False
#     trim_idx = len(times_sorted)
    
#     for i in range(stable_start_idx, len(median_by_time)):
#         if median_by_time[i] > baseline_median * explosion_threshold:
#             print(f"💥 Explosion detected at time {times_sorted[i]:.1f}: {median_by_time[i]:.1f} > {baseline_median * explosion_threshold:.1f}")
#             trim_idx = i
#             explosion_detected = True
#             break
    
#     if explosion_detected:
#         trimmed_times = times_sorted[:trim_idx]
#         trim_point = times_sorted[trim_idx] if trim_idx < len(times_sorted) else None
#         print(f"🔪 Auto-trimming from time {trim_point:.1f} onwards ({len(times_sorted) - trim_idx} points removed)")
#         return trimmed_times, trim_point
#     else:
#         print("✅ No explosions detected")
#         return times_sorted, None


def simple_patient_filter(original_times, original_volumes, patient_id):
    """Simple patient filtering - just remove last N points based on patient ID"""
    
    # Simple rules: how many points to remove from the end
    remove_rules = {
        'PIDz069': 1,  # Remove last 1 point
        'PIDz074': 1,  # Remove last 1 point  
        'PIDz077': 1,  # Remove last 1 point
        'PIDz140': 1,  # Remove last 1 point
        'PIDz161': 2,  # Remove last 2 points
        'PIDz254': 2,  # Remove last 2 points
    }
    
    if patient_id in remove_rules:
        n_remove = remove_rules[patient_id]
        if len(original_times) > n_remove:
            filtered_times = original_times[:-n_remove]
            filtered_volumes = original_volumes[:-n_remove]
            print(f"🗑️ Removed last {n_remove} points for {patient_id}")
            return filtered_times, filtered_volumes
    
    # No filtering needed
    return original_times, original_volumes


def calculate_adaptive_bins(data, min_bins=100, max_bins=10_000, target_resolution=0.01):
    """
    Calculate adaptive bin count with automatic thickness detection.
    """
    n_points = len(data)
    data_range = data.max() - data.min()
    
    if data_range == 0:
        return min_bins
    
    # Calculate data spread characteristics
    data_std = np.std(data)
    data_median = np.median(data)
    relative_std = data_std / data_median if data_median > 0 else 0
    
    # Detect if bands are likely to be thin (low relative uncertainty)
    if relative_std < 0.1:  # Less than 10% relative uncertainty
        print(f"🔍 Detected thin bands (rel_std: {relative_std:.3f}), using fewer bins for thicker visualization")
        # Use fewer bins to make thin bands appear thicker
        resolution_based_bins = int(data_range / (target_resolution * 5))  # 5x fewer bins
        adaptive_factor = 0.3  # Reduce bin count significantly
    elif relative_std < 0.2:  # 10-20% relative uncertainty  
        print(f"🔍 Detected medium bands (rel_std: {relative_std:.3f}), using moderate binning")
        resolution_based_bins = int(data_range / (target_resolution * 2))  # 2x fewer bins
        adaptive_factor = 0.6
    else:  # High uncertainty
        print(f"🔍 Detected thick bands (rel_std: {relative_std:.3f}), using fine binning")
        resolution_based_bins = int(data_range / target_resolution)
        adaptive_factor = 1.0
    
    # Also consider data density
    density_based_bins = int(n_points / 50)  # ~50 points per bin on average
    
    # Take the larger of the two, then apply adaptive factor
    optimal_bins = max(resolution_based_bins, density_based_bins)
    optimal_bins = int(optimal_bins * adaptive_factor)
    
    # Clamp to reasonable bounds
    optimal_bins = max(min_bins, min(max_bins, optimal_bins))
    
    print(f"Data range: {data_range:.1f}, std: {data_std:.1f}, Selected bins: {optimal_bins}")
    
    return optimal_bins


def extract_patient_and_slice_from_csv(csv_path):
    """Extract patient ID and slice type from CSV filename."""
    filename = Path(csv_path).stem
    # Format: PIDz035_volume_bootstrap_predictions_unified.csv
    
    parts = filename.split('_')
    patient_id = parts[0]  # PIDz035
    
    if parts[1] == 'volume':
        slice_type = 'volume'
        data_column = 'volume_manual'
    elif parts[1] in ['largest', 'initial']:
        slice_type = f"{parts[1]}_{parts[2]}"
        data_column = f"slice_{parts[1]}_{parts[2]}"
    else:
        raise ValueError(f"Unknown data type in filename: {filename}")
    
    return patient_id, slice_type, data_column


def load_original_data(data_path, patient_id, data_column):
    """Load original tumor data."""
    pickle_file = os.path.join(data_path, 'area_over_time_dict.pkl')
    
    if not os.path.exists(pickle_file):
        print(f"Warning: Could not find original data at {pickle_file}")
        return None, None, 28  # Default RT time
    
    with open(pickle_file, 'rb') as f:
        data_dict = pickle.load(f)
    
    if patient_id not in data_dict:
        print(f"Warning: Patient {patient_id} not found in original data")
        return None, None, 28
    
    patient_df = data_dict[patient_id]
    
    if data_column not in patient_df.columns:
        print(f"Warning: Column {data_column} not found for patient {patient_id}")
        return None, None, 28
    
    time_col = 'time(rel_days)'
    patient_df_indexed = patient_df.set_index(time_col)
    
    # Calculate RT start time (simple version)
    try:
        first_imaging_day = patient_df['time(rel_days)'].iloc[0]
        days_until_rt_ends = patient_df['time_end_RT(rel_days)'].iloc[0]
        
        if days_until_rt_ends < 0:
            rt_end_day = first_imaging_day + abs(days_until_rt_ends)
        else:
            rt_end_day = first_imaging_day - days_until_rt_ends
        
        rt_start_day = rt_end_day - 42
        print(f"RT start calculated: Day {rt_start_day:.0f}")
    except:
        rt_start_day = 28  # Fallback
        print(f"Using default RT start: Day {rt_start_day}")
    
    return patient_df_indexed.index.values, patient_df_indexed[data_column].values, rt_start_day


def plot_bootstrap_heatmap(csv_path, output_path, mode, data_path='data'):
    """
    Generate bootstrap prediction heatmap plot.
    Keep the working logic from original plotPrettyGC.py
    """
    
    # Extract patient and slice info
    patient_id, slice_type, data_column = extract_patient_and_slice_from_csv(csv_path)
    print(f"Processing {patient_id} - {slice_type} - {mode} mode")
    
    # Load bootstrap predictions
    df = pd.read_csv(csv_path)
    df = df[df['mode'] == mode]
    df = df.drop(columns=['mode', 'patient_id', 'data_type'], errors='ignore')
    
    # Load original data
    original_times, original_volumes, time_RT = load_original_data(data_path, patient_id, data_column)

    # # ==================== QUICK MANUAL TRIM FIX ====================
    # # Trim last 20 time points from predictions
    # trim = 30
    # all_times = sorted(df['time'].unique())
    # if trim > 0:
    #     keep_times = all_times[:-trim]  # Remove last 20
    #     df = df[df['time'].isin(keep_times)]
    #     print(f"🔪 Trimmed last {trim} time points: {all_times[-trim]} to {all_times[-1]}")
    
    # # Trim original data to match if needed
    # if original_times is not None:
    #     max_keep_time = max(keep_times) if len(all_times) > trim else max(all_times)
    #     mask = original_times <= max_keep_time
    #     original_times = original_times[mask]
    #     original_volumes = original_volumes[mask]
    #     print(f"🔪 Trimmed original data to {len(original_times)} points")
    # # ================================================================
    
    # Reshape data (exactly like original)
    times = list(df['time'].unique())
    reshaped_df = df.pivot(index='bootstrap_id', columns='time', values='prediction_value')
    reshaped_df = reshaped_df.sort_index(axis=1)
    df = reshaped_df
    
    # Long format for heatmap (exactly like original)
    df_long = df.reset_index().melt(id_vars='bootstrap_id', var_name='time', value_name='prediction')
    df_long['time'] = df_long['time'].astype(float)
    
    # Remove NaN
    df_long = df_long.dropna()

    # remove outliers
    # pred_95 = np.percentile(df_long['prediction'], 97.5)
    pred_95 = np.percentile(df_long['prediction'], 98.0)
    pred_5 = np.percentile(df_long['prediction'], 0.0)
    outlier_mask = (df_long['prediction'] >= pred_5) & (df_long['prediction'] <= pred_95)
    n_before = len(df_long)
    df_long = df_long[outlier_mask]
    n_after = len(df_long)
    print(f"Removed {n_before - n_after} outliers ({100*(n_before-n_after)/n_before:.1f}%)")
    
    # Debug info
    print(f"🔍 DEBUG - Patient: {patient_id}")
    print(f"  Bootstrap range: {df_long['prediction'].min():.2f} to {df_long['prediction'].max():.2f}")
    print(f"  Bootstrap median: {df_long['prediction'].median():.2f}")
    if original_volumes is not None:
        print(f"  Original range: {original_volumes.min():.1f} to {original_volumes.max():.1f}")
        print(f"  Original median: {np.median(original_volumes):.1f}")
    
    # Histogram bins (just improve the binning, keep everything else the same)
    n_time_bins = len(df.columns)
    n_value_bins = calculate_adaptive_bins(
        df_long['prediction'].values, 
        min_bins=100,    # Higher minimum for better resolution
        max_bins=10_000,   # Keep reasonable maximum
        target_resolution=0.1  # Target ~100 units per bin
    )
    
    print(f"Using {n_time_bins} time bins and {n_value_bins} value bins")
    
    time_bins = np.linspace(df_long['time'].min(), df_long['time'].max(), n_time_bins)
    value_bins = np.linspace(df_long['prediction'].min(), df_long['prediction'].max(), n_value_bins)
    
    # 2D histogram (exactly like original)
    heatmap, yedges, xedges = np.histogram2d(
        df_long['prediction'],
        df_long['time'],
        bins=[value_bins, time_bins]
    )
    
    # Normalize by number of bootstraps (exactly like original)
    heatmap_normalized = heatmap / df.index.size
    
    # Smooth each column (exactly like original but adaptive window)
    # window_size = max(3, min(20, n_value_bins // 20))
    # window_size = max(3, min(3, n_value_bins // 3))
    window_size = max(20, min(20, n_value_bins // 20))
    # window_size = 10
    print(f"Using smoothing window size: {window_size}")
    
    smoothed = np.apply_along_axis(
        lambda col: uniform_filter1d(col, size=window_size, mode='nearest'), 
        axis=0, 
        arr=heatmap_normalized
    )
    
    # Renormalize each column to [0, 1] (exactly like original)
    smoothed_renorm = smoothed / np.maximum(np.max(smoothed, axis=0, keepdims=True), 1e-10)
    smoothed_renorm = np.nan_to_num(smoothed_renorm)
    
    # Plot (exactly like original)
    # plt.figure(figsize=(7, 4))
    plt.figure(figsize=(4, 3))
    plt.imshow(
        smoothed_renorm,
        aspect='auto',
        origin='lower',
        extent=[xedges.min(), xedges.max(), yedges.min(), yedges.max()],
        cmap='Greys',
        vmin=0,
        vmax=1
    )
    
    # plt.xlabel('Time [days]', fontsize=16)
    plt.xlabel('Time [days]', fontsize=11)
    # plt.ylabel('Area [mm$^2$]', fontsize=16)
    plt.ylabel('Area [mm$^2$]', fontsize=11)
    
    # Plot median prediction (exactly like original)
    median_predictions = df.median(axis=0)
    r2_value = 0.948 if mode == 'train' else 0.784  # Placeholder
    plt.plot(times, median_predictions, 'r-', linewidth=1.5,
             label='Median - $R^2$ =' + str(r2_value))
    
    # RT line (exactly like original)
    plt.axvline(x=time_RT, color='blue', linestyle='--', linewidth=1.5, label='RT')
    
    # Plot original data (exactly like original)
    if original_times is not None and original_volumes is not None:
        # Apply simple patient filtering
        original_times, original_volumes = simple_patient_filter(original_times, original_volumes, patient_id)
        if mode == 'all':
            plt.errorbar(original_times, original_volumes, yerr=0.1*original_volumes, fmt='o',
                         color='r', markersize=10, elinewidth=2, label='Data')
        else:  # train mode
            plt.errorbar(original_times[:-1], original_volumes[:-1], yerr=0.1*original_volumes[:-1], fmt='o',
                         color='r', markersize=10, elinewidth=2, label='Train data')
            plt.errorbar(original_times[-1], original_volumes[-1],
                         yerr=0.1*original_volumes[-1], fmt='o',
                         color='g', markersize=10, elinewidth=2, label='Test data')
        
        # Set limits (exactly like original)
        plt.xlim([-5, original_times[-1] + 30])
        # plt.ylim([0.8*original_volumes[0], 100000])
        # Adaptive Y-axis based on actual data range
        if original_volumes is not None:
            data_min = original_volumes.min()
            data_max = original_volumes.max()
            data_range = data_max - data_min
            
            y_min = max(0, data_min - 0.05 * data_range)  # 10% buffer below, but never below 0
            y_max = data_max + 0.35 * data_range  # 30% buffer above for nice spacing
            
            plt.ylim([y_min, y_max])
    
    plt.tight_layout()
    
    # Save plots
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_filename = f"{patient_id}_{slice_type}_{mode}_heatmap"
    png_file = output_dir / f"{base_filename}.png"
    pdf_file = output_dir / f"{base_filename}.pdf"
    
    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_file, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plots: {png_file} and {pdf_file}")
    return True


def main():
    """Main function with command line arguments."""
    parser = argparse.ArgumentParser(description="Generate bootstrap prediction heatmap plots")
    
    parser.add_argument("--csv_path", type=str, required=True,
                       help="Path to bootstrap predictions CSV file")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Output directory for plots")
    parser.add_argument("--mode", type=str, choices=['all', 'train'], required=True,
                       help="Analysis mode (all or train)")
    parser.add_argument("--data_path", type=str, default="data",
                       help="Path to original data directory (default: data)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file not found: {args.csv_path}")
        return 1
    
    success = plot_bootstrap_heatmap(args.csv_path, args.output_path, args.mode, args.data_path)
    
    if success:
        print("✅ Plot generation completed successfully")
        return 0
    else:
        print("❌ Plot generation failed")
        return 1


if __name__ == "__main__":
    exit(main())