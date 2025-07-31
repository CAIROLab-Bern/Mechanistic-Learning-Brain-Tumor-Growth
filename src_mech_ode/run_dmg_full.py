#!/usr/bin/env python3
"""
Ultra-simplified DMG tumor growth model fitting - Dual Mode (Prediction + Estimation)
Minimal functions, no unnecessary classes, maximum clarity
"""
import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from utils.config import PATIENT_CONFIG
from utils.dataloading import *
from utils.model import *
from utils.export_utils import *
from utils.plots import *
from utils.predict import *
from utils.fitting import *

##############################
## Configuration & Logging
##############################
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configure matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'

##############################
## Main Analysis
##############################
def main():
    """Main analysis function."""
    
    parser = argparse.ArgumentParser(description="Simplified DMG Tumor Growth Analysis - Dual Mode")
    
    parser.add_argument("--data_path", type=str, default="data/area_over_time_dict.pkl", 
                        help="Path to pickle file with tumor data")
    parser.add_argument("--patient", type=str, 
                        help="Patient ID to analyze - if not specified, analyzes all valid patients")
    parser.add_argument("--data_type", type=str, default="volume",
                        help="Data type: volume, initial_X, largest_X where X is slice ID")
    parser.add_argument("--noise_level", type=float, default=0.1,
                        help="Noise level for bootstrap analysis - default 0.1 means 10 percent noise")
    parser.add_argument("--n_bootstrap", type=int, default=1,
                        help="Number of bootstrap samples - default 1 means no bootstrap")
    parser.add_argument("--n_jobs", type=int, default=-1,
                        help="Number of parallel workers for bootstrap - default -1 uses all cores")
    parser.add_argument("--time_column", type=str, default="time(rel_days)",
                        help="Time column name")
    parser.add_argument("--normalization", action="store_true", default=True,
                        help="Normalize data to first time point")
    parser.add_argument("--no_volume", action="store_true", default=False,
                        help="Discard initial volume param from the ODE")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory")
    parser.add_argument("--bootstrap_path", type=str, default="results_dmg_fit",
                        help="Path to existing bootstrap results for parameter bounds")
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting Simplified DMG Analysis - DUAL MODE")
    logger.info(f"Noise level: {args.noise_level}")
    logger.info(f"Bootstrap samples: {args.n_bootstrap}")
    logger.info("Running both PREDICTION and ESTIMATION modes")
    
    # Load data
    try:
        tumor_data = load_data(args.data_path, logger)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return 1
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Determine patients to analyze
    if args.patient:
        if args.patient not in tumor_data:
            logger.error(f"Patient {args.patient} not found in data")
            return 1
        if args.patient not in PATIENT_CONFIG or not PATIENT_CONFIG[args.patient]['valid']:
            logger.error(f"Patient {args.patient} not configured for analysis")
            return 1
        patients_to_analyze = [args.patient]
    else:
        # Analyze all valid patients
        all_patients = list(tumor_data.keys())
        valid_patients = [p for p in all_patients 
                         if p in PATIENT_CONFIG and PATIENT_CONFIG[p]['valid']]
        patients_to_analyze = valid_patients
        logger.info(f"Found {len(valid_patients)}/{len(all_patients)} valid patients")
    
    # Analyze patients
    all_results = []
    successful_analyses = 0
    
    for i, patient_id in enumerate(patients_to_analyze):
        logger.info(f"\n📊 Processing {patient_id} ({i+1}/{len(patients_to_analyze)})")
        
        try:
            patient_df = tumor_data[patient_id]
            results = analyze_patient_dual_mode(patient_id, patient_df, args, logger)
            
            if not results.empty:
                all_results.append(results)
                successful_analyses += 1
                logger.info(f"✅ Successfully analyzed {patient_id}")
            else:
                logger.warning(f"❌ Failed to analyze {patient_id}")
                
        except Exception as e:
            logger.error(f"❌ Error analyzing {patient_id}: {e}")
            continue
    
    # Generate summary
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # Save combined results
        summary_file = Path(args.output_dir) / f"summary_dual_mode_{args.data_type}.csv"
        combined_results.to_csv(summary_file, index=False)
        
        # Generate summary statistics
        summary_stats_file = Path(args.output_dir) / f"summary_stats_dual_mode_{args.data_type}.txt"
        with open(summary_stats_file, 'w') as f:
            f.write("Simplified DMG Analysis Summary - DUAL MODE\n")
            f.write("==========================================\n")
            f.write(f"Data type: {args.data_type}\n")
            f.write(f"Noise level: {args.noise_level}\n")
            f.write(f"Bootstrap samples: {args.n_bootstrap}\n")
            f.write(f"Patients processed: {len(patients_to_analyze)}\n")
            f.write(f"Successful analyses: {successful_analyses}\n")
            f.write(f"Success rate: {successful_analyses/len(patients_to_analyze)*100:.1f}%\n\n")
            
            # Prediction mode statistics
            if 'prediction_r2' in combined_results.columns:
                pred_r2_values = combined_results['prediction_r2'].dropna()
                if len(pred_r2_values) > 0:
                    f.write("PREDICTION MODE R² Statistics:\n")
                    f.write(f"  Mean: {pred_r2_values.mean():.3f}\n")
                    f.write(f"  Median: {pred_r2_values.median():.3f}\n")
                    f.write(f"  Std: {pred_r2_values.std():.3f}\n")
                    f.write(f"  Min: {pred_r2_values.min():.3f}\n")
                    f.write(f"  Max: {pred_r2_values.max():.3f}\n\n")
            
            # Estimation mode statistics
            if 'estimation_train_r2' in combined_results.columns:
                est_train_r2_values = combined_results['estimation_train_r2'].dropna()
                if len(est_train_r2_values) > 0:
                    f.write("ESTIMATION MODE Train R² Statistics:\n")
                    f.write(f"  Mean: {est_train_r2_values.mean():.3f}\n")
                    f.write(f"  Median: {est_train_r2_values.median():.3f}\n")
                    f.write(f"  Std: {est_train_r2_values.std():.3f}\n")
                    f.write(f"  Min: {est_train_r2_values.min():.3f}\n")
                    f.write(f"  Max: {est_train_r2_values.max():.3f}\n\n")
            
            if 'estimation_test_rmse' in combined_results.columns:
                est_rmse_values = combined_results['estimation_test_rmse'].dropna()
                if len(est_rmse_values) > 0:
                    f.write("ESTIMATION MODE Test RMSE Statistics:\n")
                    f.write(f"  Mean: {est_rmse_values.mean():.3f}\n")
                    f.write(f"  Median: {est_rmse_values.median():.3f}\n")
                    f.write(f"  Std: {est_rmse_values.std():.3f}\n")
                    f.write(f"  Min: {est_rmse_values.min():.3f}\n")
                    f.write(f"  Max: {est_rmse_values.max():.3f}\n\n")
        
        logger.info("\n🎉 Analysis completed successfully!")
        logger.info(f"📊 Results saved to: {args.output_dir}")
        logger.info(f"📈 Summary: {summary_file}")
        logger.info(f"📋 Stats: {summary_stats_file}")
        logger.info(f"Success rate: {successful_analyses}/{len(patients_to_analyze)} patients")
        
        # Print quick summary
        if 'prediction_r2' in combined_results.columns:
            pred_r2_values = combined_results['prediction_r2'].dropna()
            if len(pred_r2_values) > 0:
                logger.info(f"Prediction R² summary: {pred_r2_values.mean():.3f} ± {pred_r2_values.std():.3f} "
                           f"(range: {pred_r2_values.min():.3f} - {pred_r2_values.max():.3f})")
        
        if 'estimation_train_r2' in combined_results.columns:
            est_train_r2_values = combined_results['estimation_train_r2'].dropna()
            if len(est_train_r2_values) > 0:
                logger.info(f"Estimation Train R² summary: {est_train_r2_values.mean():.3f} ± {est_train_r2_values.std():.3f} "
                           f"(range: {est_train_r2_values.min():.3f} - {est_train_r2_values.max():.3f})")
        
        if 'estimation_test_rmse' in combined_results.columns:
            est_rmse_values = combined_results['estimation_test_rmse'].dropna()
            if len(est_rmse_values) > 0:
                logger.info(f"Estimation Test RMSE summary: {est_rmse_values.mean():.3f} ± {est_rmse_values.std():.3f} "
                           f"(range: {est_rmse_values.min():.3f} - {est_rmse_values.max():.3f})")
    
    else:
        logger.error("❌ No patients were successfully analyzed")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())