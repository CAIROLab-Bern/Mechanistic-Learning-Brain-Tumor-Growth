"""
Model fitting and bootstrap analysis for DMG tumor growth analysis.
"""
import multiprocessing as mp
from typing import Dict, List, Tuple
import logging

import lmfit
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import r2_score
from tqdm import tqdm

from .model import (
    solve_tumor_dynamics, objective_function, 
    create_fitting_parameters_from_bootstrap
)
from .helpers import clean_predictions_array

def fit_patient_data(
        data_dict: Dict,
        patient_id: str, 
        noise_level: float = 0.1,
        no_volume: bool = False,
        slice_id: str = "volume",
        mode: str = "all",
        bootstrap_path: str = "results_dmg_fit",
        logger: logging.Logger = None,
    ) -> Tuple[pd.DataFrame, Dict]:
    """Fit model to patient data"""
    
    if logger:
        logger.info(f"Fitting patient {patient_id} with noise level {noise_level}")
    
    initial_volume = data_dict['v_1'].iloc[0]
    parameters = create_fitting_parameters_from_bootstrap(
        bootstrap_path, patient_id, slice_id,
        mode, initial_volume, noise_level, no_volume
    )
    
    # Fit model with least squares
    result = lmfit.minimize(
        objective_function,
        parameters,
        args=(data_dict, no_volume,),
        method="leastsq",
        nan_policy="omit",
        max_nfev=10000,
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
    )
    
    if not result.success and logger:
        logger.warning(f"Optimization failed for {patient_id}: {result.message}")
    
    fitted_params = {name: param.value for name, param in result.params.items()}
    
    # Calculate R²
    time_data = data_dict['v_1'].index.values
    volume_data = data_dict['v_1'].values
    rt_time = data_dict['rt_time']
    
    model_solution = solve_tumor_dynamics(time_data, fitted_params, rt_time, fitted_params['SD_1'], no_volume)
    model_volume = model_solution.sum(axis=1)
    
    r2 = r2_score(volume_data, model_volume)
    
    # Create results dataframe
    results_df = pd.DataFrame()
    results_df.loc[patient_id, 'R2_1'] = r2
    
    for param_name, param_value in fitted_params.items():
        results_df.loc[patient_id, param_name] = param_value
    
    results_df.loc[patient_id, 'rt_time'] = rt_time
    results_df.loc[patient_id, 'noise_level'] = noise_level
    
    if logger:
        logger.info(f"Patient {patient_id}: R² = {r2:.3f}")
    
    return results_df, fitted_params

def single_bootstrap_fit(args):
    """Single bootstrap iteration for parallel processing."""
    seed, data_dict, patient_id, noise_level, no_volume, data_type, mode, bootstrap_path, logger = args
    
    try:
        np.random.seed(seed)
        
        # Create bootstrap sample by adding noise
        bootstrap_data = data_dict.copy()
        
        original_volumes = data_dict['v_1']
        noise_multipliers = np.random.normal(1.0, noise_level, len(original_volumes))
        noise_multipliers = np.clip(noise_multipliers, 1e-3, 1.0)
        
        noisy_volumes = original_volumes * noise_multipliers
        noisy_volumes = np.maximum(noisy_volumes, 0.05)

        # CLEAN THE NOISY VOLUMES BEFORE FITTING
        noisy_volumes = clean_predictions_array(noisy_volumes, f"{patient_id}_bs{seed}", logger, allow_negative=False)

        # CHECK IF ALL VALUES ARE ZERO AFTER CLEANING
        if np.all(noisy_volumes == 0) or len(noisy_volumes) == 0:
            if logger:
                logger.warning(f"Bootstrap sample {seed} has all zero values after cleaning")
            return None
        
        # ENSURE MINIMUM VARIATION FOR V_01 PARAMETER
        if np.std(noisy_volumes) < 1e-6:
            if logger:
                logger.warning(f"Bootstrap sample {seed} has insufficient variation")
            return None
        
        bootstrap_data['v_1'] = pd.Series(noisy_volumes, index=original_volumes.index)
        
        # Validate bootstrap data
        if len(bootstrap_data['v_1']) < 3:
            return None
        
        # Fit bootstrap sample
        results_df, fitted_params = fit_patient_data(
            bootstrap_data,
            patient_id,
            noise_level,
            no_volume,
            data_type,
            mode,
            bootstrap_path,
            logger,
        )
        
        if results_df.empty:
            return None
            
        result_row = results_df.iloc[0]
        
        # Check for valid parameters
        if result_row.isnull().any():
            return None
        
        # Return both result row and fitted_params for prediction generation
        return {'result_row': result_row, 'fitted_params': fitted_params}
        
    except Exception as e:
        if logger:
            logger.warning(f"Bootstrap sample {seed} failed: {e}")
        return None

def bootstrap_analysis(
        data_dict: Dict,
        patient_id: str,
        n_bootstrap: int = 50, 
        noise_level: float = 0.1,
        n_jobs: int = -1,
        no_volume: bool = False,
        data_type: str = "volume",
        mode: str = "all",
        bootstrap_path: str = "results_dmg_fit",
        logger: logging.Logger = None,
    ) -> Tuple[pd.DataFrame, np.ndarray, List[Dict]]:
    """Run bootstrap analysis with parallel processing."""
    
    if logger:
        logger.info(f"Running bootstrap analysis: {n_bootstrap} samples, noise level: {noise_level}")
    
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    elif n_jobs <= 0:
        n_jobs = 1
    
    if logger:
        logger.info(f"Using {n_jobs} parallel workers for bootstrap")
    
    # Prepare arguments for parallel processing
    args_list = \
        [(i, data_dict, patient_id, noise_level, no_volume, data_type, mode, bootstrap_path, logger) for i in range(n_bootstrap)]
    
    # Run parallel bootstrap
    if n_jobs > 1:
        bootstrap_results_list = Parallel(n_jobs=n_jobs, backend='multiprocessing')(
            delayed(single_bootstrap_fit)(args) for args in args_list
        )
    else:
        bootstrap_results_list = [single_bootstrap_fit(args) for args in tqdm(args_list, desc="Bootstrap")]
    
    # Filter successful results
    successful_results = [r for r in bootstrap_results_list if r is not None]
    
    if not successful_results:
        if logger:
            logger.error("All bootstrap iterations failed")
        return pd.DataFrame(), np.empty((0, 0, 0)), []
    
    if logger:
        logger.info(f"Successful bootstrap iterations: {len(successful_results)}/{n_bootstrap}")
    
    # Extract result rows and fitted parameters
    bootstrap_results = pd.DataFrame([r['result_row'] for r in successful_results])
    fitted_params_list = [r['fitted_params'] for r in successful_results]
    
    # Generate prediction envelope
    all_times = data_dict['v_1'].index.values
    t_min, t_max = all_times.min(), all_times.max()
    t_plot = np.linspace(t_min, t_max, int(t_max - t_min) + 1)
    
    prediction_storage = np.empty((1, len(t_plot), len(successful_results)))
    prediction_storage[:] = np.nan
    
    # Generate predictions for each successful bootstrap result
    for idx, fitted_params in enumerate(fitted_params_list):
        try:
            rt_time = data_dict['rt_time']
            
            model_solution = solve_tumor_dynamics(t_plot, fitted_params, rt_time, fitted_params['SD_1'], no_volume)
            prediction_storage[0, :, idx] = model_solution.sum(axis=1)
            
        except Exception as e:
            if logger:
                logger.warning(f"Failed to generate predictions for bootstrap {idx}: {e}")
            continue
    
    return bootstrap_results, prediction_storage, fitted_params_list

def run_all_mode(
        data_dict: Dict, patient_id: str, data_type: str, 
        output_dir, args, logger: logging.Logger
    ) -> Tuple[Dict, pd.DataFrame, np.ndarray]:
    """Run all mode analysis."""
    logger.info(f"🔮 Running ALL mode for {patient_id}")
    
    # Import here to avoid circular imports
    from .plots import plot_model_fit_with_uncertainty, plot_parameter_distributions
    from .export_utils import export_bootstrap_parameters, export_bootstrap_predictions
    
    # Bootstrap analysis on all data
    if args.n_bootstrap > 1:
        bootstrap_results, bootstrap_predictions, fitted_params_list = bootstrap_analysis(
            data_dict,
            patient_id,
            args.n_bootstrap,
            args.noise_level,
            args.n_jobs,
            args.no_volume,
            data_type,
            "all",
            args.bootstrap_path,
            logger
        )
        
        if not bootstrap_results.empty and bootstrap_predictions.size > 0:
            # Calculate R² from median curve - USE ORIGINAL TIMELINE
            time_data = data_dict['v_1'].index.values
            volume_data = data_dict['v_1'].values * data_dict['norm_factor']
            
            # Use the SAME timeline that bootstrap_predictions was generated with
            t_min, t_max = time_data.min(), time_data.max()
            t_plot = np.linspace(t_min, t_max, bootstrap_predictions.shape[1])  # ← ORIGINAL timeline
            
            predictions_denorm = bootstrap_predictions[0, :, :] * data_dict['norm_factor']
            median_volume = np.nanmedian(predictions_denorm, axis=1)
            median_at_data_times = np.interp(time_data, t_plot, median_volume)
            r2 = r2_score(volume_data, median_at_data_times)
            
            # Create visualizations
            plot_model_fit_with_uncertainty(data_dict, bootstrap_predictions, 
                                           patient_id, output_dir, data_type, 'prediction', logger)
            
            if len(bootstrap_results) > 5:
                plot_parameter_distributions(bootstrap_results, patient_id, output_dir, data_type, logger)
            
            # Save results
            results_file = output_dir / f"{patient_id}_{data_type}_prediction_results.csv"
            bootstrap_results.to_csv(results_file)
            logger.info(f"Saved prediction results: {results_file}")

            # For export, create extended timeline for predictions if needed
            extension_days = 70
            t_max_extended = t_max + extension_days
            t_plot_extended = np.linspace(t_min, t_max_extended, int(t_max_extended - t_min) + 1)
            
            # Generate extended predictions for export
            extended_predictions = np.empty((1, len(t_plot_extended), len(fitted_params_list)))
            extended_predictions[:] = np.nan
            
            for idx, fitted_params in enumerate(fitted_params_list):
                try:
                    rt_time = data_dict['rt_time']
                    model_solution = solve_tumor_dynamics(t_plot_extended, fitted_params, rt_time, fitted_params['SD_1'], args.no_volume)
                    extended_predictions[0, :, idx] = model_solution.sum(axis=1)
                except Exception as e:
                    logger.warning(f"Failed to generate extended predictions for bootstrap {idx}: {e}")
                    continue

            # Export bootstrap data
            export_bootstrap_parameters(bootstrap_results, patient_id, data_type, 'all', output_dir, logger)
            export_bootstrap_predictions(extended_predictions, t_plot_extended, patient_id, data_type, 'all', 
                                        output_dir, data_dict['norm_factor'], logger)

            return {'r2': r2, 'n_bootstrap': len(bootstrap_results)}, bootstrap_results, bootstrap_predictions
            
    else:
        # Single fit
        results_df, fitted_params = fit_patient_data(
            data_dict,
            patient_id,
            args.noise_level,
            args.no_volume,
            data_type,
            "all",
            args.bootstrap_path,
            logger,
        )
        if not results_df.empty:
            r2 = results_df.iloc[0]['R2_1']
            
            # Save results
            results_file = output_dir / f"{patient_id}_{data_type}_prediction_results.csv"
            results_df.to_csv(results_file)
            logger.info(f"Saved prediction results: {results_file}")
            
            return {'r2': r2, 'n_bootstrap': 1}, pd.DataFrame(), np.array([])
    
    return {'r2': np.nan, 'n_bootstrap': 0}, pd.DataFrame(), np.array([])

def run_train_mode(train_data: Dict, test_data: Dict, patient_id: str, data_type: str,
                   output_dir, args, logger: logging.Logger) -> Tuple[Dict, pd.DataFrame, Dict]:
    """Run train mode analysis."""
    
    logger.info(f"📊 Running Train mode for {patient_id}")
    
    # Import here to avoid circular imports
    from .config import P_PERCENTILE
    from .predict import generate_predictions_multi_step, get_median_solution, get_prediction_t
    from .plots import plot_estimation_train_test, plot_estimation_prediction_distribution
    from .export_utils import export_bootstrap_parameters, export_bootstrap_predictions
    
    # Bootstrap analysis on training data
    if args.n_bootstrap > 1:
        bootstrap_results, _, fitted_params_list = bootstrap_analysis(
            train_data, patient_id, args.n_bootstrap, args.noise_level, args.n_jobs, 
            args.no_volume, data_type, "train", args.bootstrap_path, logger
        )
        
        if not bootstrap_results.empty and fitted_params_list:
            # Calculate train R² from median parameters
            ######### Fit
            median_params = {}
            for key in fitted_params_list[0].keys():
                values = [fp[key] for fp in fitted_params_list]
                median_params[key] = np.median(values)
            
            train_times = train_data['v_1'].index.values
            train_volumes = train_data['v_1'].values * train_data['norm_factor']
            rt_time = train_data['rt_time']
            
            train_model_solution = solve_tumor_dynamics(train_times, median_params, rt_time, median_params['SD_1'], args.no_volume)
            train_model_volumes = train_model_solution.sum(axis=1) * train_data['norm_factor']
            train_r2 = r2_score(train_volumes, train_model_volumes)
            
            ########### Predict
            # Generate test predictions
            test_times = test_data['v_1'].index.values
            test_volumes = test_data['v_1'].values * train_data['norm_factor']
            
            assert train_data['norm_factor'] == test_data['norm_factor']

            all_times = np.concatenate([train_times, test_times])  # Full timeline
            t_min, t_max = all_times.min(), all_times.max()
            extension_time = 70
            t_max_extended = t_max + extension_time
            t_plot_train = np.linspace(t_min, t_max_extended, int(t_max_extended - t_min) + 1)

            # A dict with all the solutions and the corresponding solution time-scale
            all_test_predictions = generate_predictions_multi_step(
                fitted_params_list, train_times, test_times, rt_time, train_data['norm_factor'], args.no_volume
            )
            y_median, median_prediction = get_median_solution(all_test_predictions, test_times, logger)
            
            rmse = np.sqrt(((test_volumes - median_prediction) ** 2).mean())
            
            # Create visualizations
            plot_estimation_train_test(
                train_data, test_data, all_test_predictions.copy(), 
                patient_id, output_dir, data_type, logger
            )
            plot_estimation_prediction_distribution(
                train_data, test_data, all_test_predictions.copy(), 
                patient_id, output_dir, data_type, logger
            )
            
            # get the predictions at the time of interest
            all_test_predictions_at_times = get_prediction_t(all_test_predictions, test_times, logger)
            all_test_predictions_at_times = clean_predictions_array(all_test_predictions_at_times, patient_id, logger)
            
            # Save aggregated statistics CSV
            aggregated_stats = {
                'median': np.median(all_test_predictions_at_times),
                'mean': np.mean(all_test_predictions_at_times),
                'std': np.std(all_test_predictions_at_times),
                'ci_lower': np.percentile(all_test_predictions_at_times, P_PERCENTILE),
                'ci_upper': np.percentile(all_test_predictions_at_times, 100 - P_PERCENTILE),
                'ground_truth': test_volumes[0] if len(test_volumes) == 1 else test_volumes.mean(),
                'rmse': rmse,
                'train_r2': train_r2,
                'n_bootstrap': len(all_test_predictions_at_times)
            }
            
            agg_stats_df = pd.DataFrame([aggregated_stats])
            agg_file = output_dir / f"{patient_id}_{data_type}_estimation_aggregated.csv"
            agg_stats_df.to_csv(agg_file, index=False)
            logger.info(f"Saved aggregated statistics: {agg_file}")
            
            # Save bootstrap predictions CSV
            bootstrap_df = pd.DataFrame({
                'bootstrap_id': range(len(all_test_predictions_at_times)),
                'predicted_value': all_test_predictions_at_times,
                'ground_truth': [test_volumes[0] if len(test_volumes) == 1 else test_volumes.mean()] * len(all_test_predictions_at_times),
                'test_time': test_times[0]
            })
            bootstrap_file = output_dir / f"{patient_id}_{data_type}_estimation_bootstrap.csv"
            bootstrap_df.to_csv(bootstrap_file, index=False)
            logger.info(f"Saved bootstrap predictions: {bootstrap_file}")

            export_bootstrap_parameters(bootstrap_results, patient_id, data_type, 'train', output_dir, logger)

            # Generate bootstrap predictions array for train mode
            train_predictions_storage = np.empty((1, len(t_plot_train), len(fitted_params_list)))
            train_predictions_storage[:] = np.nan

            for idx, fitted_params in enumerate(fitted_params_list):
                try:
                    model_solution = solve_tumor_dynamics(t_plot_train, fitted_params, rt_time, fitted_params['SD_1'], args.no_volume)
                    train_predictions_storage[0, :, idx] = model_solution.sum(axis=1)
                except Exception as e:
                    logger.warning(f"Failed to generate train mode predictions for bootstrap {idx}: {e}")
                    continue

            export_bootstrap_predictions(train_predictions_storage, t_plot_train, patient_id, data_type, 'train', 
                                        output_dir, train_data['norm_factor'], logger)
            
            return {
                'train_r2': train_r2, 
                'test_rmse': rmse, 
                'n_bootstrap': len(all_test_predictions_at_times),
                'median_prediction': median_prediction,
                'ground_truth': test_volumes[0] if len(test_volumes) == 1 else test_volumes.mean()
            }, bootstrap_results, all_test_predictions
    else:
        # Single fit
        results_df, fitted_params = fit_patient_data(
            train_data, patient_id, args.noise_level,
            args.no_volume, data_type, "train", args.bootstrap_path, logger
        )
        if not results_df.empty:
            train_r2 = results_df.iloc[0]['R2_1']
            
            # Test prediction
            test_times = test_data['v_1'].index.values
            test_volumes = test_data['v_1'].values * test_data['norm_factor']
            rt_time = train_data['rt_time']
            
            test_model_solution = solve_tumor_dynamics(test_times, fitted_params, rt_time, fitted_params['SD_1'], args.no_volume)
            test_model_volumes = test_model_solution.sum(axis=1) * test_data['norm_factor']
            rmse = np.sqrt(((test_volumes - test_model_volumes) ** 2).mean())
            
            return {
                'train_r2': train_r2, 
                'test_rmse': rmse, 
                'n_bootstrap': 1,
                'median_prediction': test_model_volumes[0] if len(test_model_volumes) == 1 else test_model_volumes.mean(),
                'ground_truth': test_volumes[0] if len(test_volumes) == 1 else test_volumes.mean()
            }, pd.DataFrame(), {}
    
    return {'train_r2': np.nan, 'test_rmse': np.nan, 'n_bootstrap': 0}, pd.DataFrame(), {}

def analyze_patient_dual_mode(patient_id: str, patient_df, args, logger: logging.Logger):
    """Analyze a single patient in both all and train modes."""
    
    from pathlib import Path
    from .dataloading import print_patient_slice_info, prepare_patient_data, apply_train_test_split
    from .export_utils import merge_bootstrap_exports
    
    logger.info(f"\n🔬 Analyzing patient {patient_id} - DUAL MODE")
    
    # Print available slice information
    print_patient_slice_info(patient_id, patient_df)
    
    # Prepare data
    try:
        data_dict = prepare_patient_data(
            patient_df, args.data_type, 
            args.time_column, args.normalization,
            args.noise_level
        )
    except ValueError as e:
        logger.error(f"Data preparation failed for {patient_id}: {e}")
        return pd.DataFrame()
    
    # Apply train/test split for train mode
    train_data, test_data = apply_train_test_split(data_dict, patient_id, 'estimation')
    if train_data is None:
        logger.error(f"Patient {patient_id} not configured for analysis")
        return pd.DataFrame()
    
    # Prepare data for all mode (all data)
    all_data, _ = apply_train_test_split(data_dict, patient_id, 'prediction')
    
    # Create output directories
    all_dir = Path(args.output_dir) / patient_id / 'all' / args.data_type
    train_dir = Path(args.output_dir) / patient_id / 'train' / args.data_type
    all_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    
    # Run both modes
    results_summary = {}
    
    # ALL MODE
    all_results, all_bootstrap_results, all_bootstrap_predictions = run_all_mode(
        all_data, patient_id, args.data_type, all_dir, args, logger
    )
    results_summary.update({f'all_{k}': v for k, v in all_results.items()})
    
    # TRAIN MODE
    train_results, train_bootstrap_results, train_predictions = run_train_mode(
        train_data, test_data, patient_id, args.data_type, train_dir, args, logger
    )
    results_summary.update({f'train_{k}': v for k, v in train_results.items()})
    
    # Export and merge bootstrap data (only if bootstrap was used)
    if args.n_bootstrap > 1:
        # Prepare data for merging
        all_params_df = None
        train_params_df = None
        all_preds_df = None
        train_preds_df = None
        
        # Read back the exported files for merging
        try:
            all_params_file = all_dir / f"{patient_id}_{args.data_type}_bootstrap_params_all.csv"
            if all_params_file.exists():
                all_params_df = pd.read_csv(all_params_file)
        except:
            logger.warning(f"Could not read all mode parameters for merging")
        
        try:
            train_params_file = train_dir / f"{patient_id}_{args.data_type}_bootstrap_params_train.csv"
            if train_params_file.exists():
                train_params_df = pd.read_csv(train_params_file)
        except:
            logger.warning(f"Could not read train mode parameters for merging")
        
        try:
            all_preds_file = all_dir / f"{patient_id}_{args.data_type}_bootstrap_predictions_all.csv"
            if all_preds_file.exists():
                all_preds_df = pd.read_csv(all_preds_file)
        except:
            logger.warning(f"Could not read all mode predictions for merging")
        
        try:
            train_preds_file = train_dir / f"{patient_id}_{args.data_type}_bootstrap_predictions_train.csv"
            if train_preds_file.exists():
                train_preds_df = pd.read_csv(train_preds_file)
        except:
            logger.warning(f"Could not read train mode predictions for merging")
        
        # Create unified directory and merge files
        unified_dir = Path(args.output_dir) / patient_id / 'unified' / args.data_type
        unified_dir.mkdir(parents=True, exist_ok=True)
        
        merge_bootstrap_exports(all_params_df, train_params_df, all_preds_df, train_preds_df,
                              patient_id, args.data_type, unified_dir, logger)
    
    # Create combined results dataframe
    results_df = pd.DataFrame([results_summary])
    results_df['patient_id'] = patient_id
    results_df['data_type'] = args.data_type
    results_df['noise_level'] = args.noise_level
    
    # Save combined summary
    summary_file = Path(args.output_dir) / patient_id / f"{patient_id}_{args.data_type}_dual_mode_summary.csv"
    results_df.to_csv(summary_file, index=False)
    logger.info(f"Saved dual mode summary: {summary_file}")
    
    logger.info(f"✅ {patient_id} - All R²: {all_results.get('r2', 'N/A'):.3f}, "
               f"Train R²: {train_results.get('train_r2', 'N/A'):.3f}, "
               f"Test RMSE: {train_results.get('test_rmse', 'N/A'):.3f}")
    
    return results_df