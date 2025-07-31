"""
Model definitions and fitting utilities for DMG tumor growth analysis.
"""
import os
import copy
import warnings
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional
import logging

import lmfit
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.integrate import odeint
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

from .helpers import clean_predictions_array

def ode_system(state: List[float], time: float, lambda_gr: float, lambda_decay: float, 
               delay: float, slope: float, rt_time: float) -> List[float]:
    """Simple 5-compartment tumor ODE system."""
    V_l, Q_l, D, V_d, Q_d = state
    
    time_since_rt = time - delay - rt_time
    lambda_decay_active = -np.tanh(slope * time_since_rt) * lambda_decay
    
    dV_l_dt = lambda_gr * V_l
    dQ_l_dt = 0.0
    dD_dt = 0.0
    
    # Prevent overflow
    if abs(lambda_decay_active) < 1e-10 or abs(V_d) < 1e-10:
        dV_d_dt = 0.0
    else:
        decay_product = lambda_decay_active * V_d
        dV_d_dt = np.clip(decay_product, -1e10, 1e10)
    
    dQ_d_dt = 0.0
    
    return [dV_l_dt, dQ_l_dt, dD_dt, dV_d_dt, dQ_d_dt]

def solve_tumor_dynamics(
        time_points: np.ndarray, params: Dict[str, float], 
        rt_time: float, survival_fraction: float,
        no_volume: Optional[bool] = False,
    ) -> np.ndarray:
    """Solve tumor growth with treatment effects."""
    
    # Initial conditions
    if no_volume:
        initial_volume = 1.0
    else:
        initial_volume = params['V_01']
    initial_conditions = [initial_volume, 0.0, 0.0, 0.0, 0.0]
    
    # Pre-RT phase
    pre_rt_times = time_points[time_points <= rt_time]
    if len(pre_rt_times) == 0 or pre_rt_times[-1] < rt_time:
        pre_rt_times = np.append(pre_rt_times, rt_time)
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        
        pre_rt_solution = odeint(
            ode_system, initial_conditions, pre_rt_times,
            args=(params['lambda_gr'], params['lambda_decay'], 
                  params['delay'], params['slope'], rt_time),
            rtol=1e-4, atol=1e-6, mxstep=50000
        )
    
    # Apply treatment effect
    state_before_rt = pre_rt_solution[-1, :]
    state_after_rt = [
        state_before_rt[0] * survival_fraction,
        state_before_rt[1] * survival_fraction,
        state_before_rt[2],
        state_before_rt[3] + (1 - survival_fraction) * state_before_rt[0],
        state_before_rt[4] + (1 - survival_fraction) * state_before_rt[1],
    ]
    
    # Post-RT phase
    post_rt_times = time_points[time_points > rt_time]
    if len(post_rt_times) > 0:
        post_rt_times_with_start = np.concatenate([[rt_time], post_rt_times])
        
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                
                post_rt_solution = odeint(
                    ode_system, state_after_rt, post_rt_times_with_start,
                    args=(params['lambda_gr'], params['lambda_decay'], 
                          params['delay'], params['slope'], rt_time),
                    rtol=1e-4, atol=1e-6, mxstep=50000
                )
            post_rt_solution = post_rt_solution[1:, :]
        except:
            post_rt_solution = np.tile(state_after_rt, (len(post_rt_times), 1))
    else:
        post_rt_solution = np.empty((0, 5))
    
    # Combine solutions
    full_solution = np.zeros((len(time_points), 5))
    
    pre_rt_mask = time_points <= rt_time
    if np.any(pre_rt_mask):
        pre_rt_indices = np.where(pre_rt_mask)[0]
        if len(pre_rt_indices) <= len(pre_rt_solution) - 1:
            full_solution[pre_rt_indices, :] = pre_rt_solution[:-1, :]
        else:
            full_solution[pre_rt_indices, :] = pre_rt_solution[:len(pre_rt_indices), :]
    
    post_rt_mask = time_points > rt_time
    if np.any(post_rt_mask):
        post_rt_indices = np.where(post_rt_mask)[0]
        if len(post_rt_indices) == len(post_rt_solution):
            full_solution[post_rt_indices, :] = post_rt_solution
    
    return full_solution

def objective_function(parameters: lmfit.Parameters, data_dict: Dict, no_volume: Optional[bool] = False) -> np.ndarray:
    """Objective function for parameter fitting."""
    
    param_dict = {name: param.value for name, param in parameters.items()}
    
    # Validate parameters
    for param_name, param_value in param_dict.items():
        if not np.isfinite(param_value):
            return np.array([1e6] * 10)
    
    try:
        volume_data = data_dict['v_1'].values
        time_data = data_dict['v_1'].index.values
        rt_time = data_dict['rt_time']
        survival_fraction = param_dict['SD_1']
        
        model_solution = solve_tumor_dynamics(time_data, param_dict, rt_time, survival_fraction, no_volume)
        model_volume = model_solution.sum(axis=1)
        
        residuals = (volume_data - model_volume) ** 2
        return np.clip(residuals, 0, 1e12)
        
    except Exception as e:
        # Note: logger will be passed in later when we refactor this function
        return np.array([1e6] * 10)

def create_fitting_parameters(initial_volume: float, noise_level: float = 0.1, no_volume: bool = False) -> lmfit.Parameters:
    """Create fitting parameters with noise level consideration."""
    
    params = lmfit.Parameters()
    
    # Add noise-dependent bounds - higher noise allows wider parameter exploration
    noise_factor = 1 + noise_level
    
    params.add("lambda_gr", value=0.02, min=0.001/noise_factor, max=2.0*noise_factor)
    params.add("lambda_decay", value=0.02, min=0.001/noise_factor, max=100.0*noise_factor)
    params.add("delay", value=5.0, min=0.001, max=300.0*noise_factor)
    params.add("slope", value=1.0, min=0.001, max=600*noise_factor)
    params.add("SD_1", value=0.0001, min=0.001, max=0.20*noise_factor)
    
    if no_volume:
        print("No Volume param included")
    else:
        params.add("V_01", value=initial_volume, 
                min=initial_volume * (1-noise_level), 
                max=initial_volume * (1+noise_level))
    
    return params

def create_fitting_parameters_from_bootstrap(csv_path: str, patient_id: str, slice_id: str, mode: str, 
                                           initial_volume: float, noise_level: float = 0.1, 
                                           no_volume: bool = False) -> lmfit.Parameters:
    """
    Create fitting parameters with bounds based on bootstrap distribution analysis.
    Falls back to default parameters if bootstrap data is not available.
    """
    # Build path to bootstrap data
    total_csv_path = os.path.join(
        csv_path, patient_id, "unified", slice_id,
        f"{patient_id}_{slice_id}_bootstrap_params_unified.csv"
    )
    
    # Check if file exists
    if not os.path.exists(total_csv_path):
        print(f"⚠️  Bootstrap file not found: {total_csv_path}")
        print(f"⚠️  Using default parameter bounds for {patient_id}")
        return create_fitting_parameters(initial_volume, noise_level, no_volume)
    
    try:
        # Read and filter bootstrap data
        df = pd.read_csv(total_csv_path)
        filtered_df = df[
            (df['patient_id'] == patient_id) & 
            (df['data_type'] == slice_id) & 
            (df['mode'] == mode)
        ]
        
        if len(filtered_df) == 0:
            print(f"⚠️  No matching data found for {patient_id}, {slice_id}, {mode} - using default bounds")
            return create_fitting_parameters(initial_volume, noise_level, no_volume)
        
        # Remove top 10% for each parameter INDEPENDENTLY
        lambda_gr_90th = np.percentile(filtered_df['lambda_gr'], 90)
        lambda_decay_90th = np.percentile(filtered_df['lambda_decay'], 90)
        delay_90th = np.percentile(filtered_df['delay'], 90)
        
        # Get max values from the remaining 90% for each parameter separately
        lambda_gr_trimmed = filtered_df[filtered_df['lambda_gr'] <= lambda_gr_90th]['lambda_gr']
        lambda_decay_trimmed = filtered_df[filtered_df['lambda_decay'] <= lambda_decay_90th]['lambda_decay']
        delay_trimmed = filtered_df[filtered_df['delay'] <= delay_90th]['delay']
        
        max_lambda_gr = lambda_gr_trimmed.max()
        max_lambda_decay = lambda_decay_trimmed.max()
        max_delay = delay_trimmed.max()
        
        print(f"📊 Bootstrap-based bounds for {patient_id}:")
        print(f"  lambda_gr: max = {max_lambda_gr:.4f} (from {len(lambda_gr_trimmed)}/{len(filtered_df)} samples)")
        print(f"  lambda_decay: max = {max_lambda_decay:.4f} (from {len(lambda_decay_trimmed)}/{len(filtered_df)} samples)")
        print(f"  delay: max = {max_delay:.4f} (from {len(delay_trimmed)}/{len(filtered_df)} samples)")
        print(f" ----$$$$$$$$$$$$$$$----")
        
        # Create parameters with adaptive bounds
        params = lmfit.Parameters()
        noise_factor = 1 + noise_level
        
        # Use bootstrap-informed maximum bounds
        params.add("lambda_gr", value=0.02, min=0.001/noise_factor, max=max_lambda_gr*noise_factor)
        params.add("lambda_decay", value=0.02, min=0.001/noise_factor, max=max_lambda_decay*noise_factor)
        params.add("delay", value=60, min=0.0, max=max_delay*noise_factor)
        
        # Keep original bounds for other parameters
        params.add("slope", value=1.0, min=0.001, max=600*noise_factor)
        params.add("SD_1", value=0.0001, min=0.001, max=0.20*noise_factor)
        
        if not no_volume:
            params.add("V_01", value=initial_volume, 
                      min=initial_volume * (1-noise_level), 
                      max=initial_volume * (1+noise_level))
        else:
            print("No Volume param included")
        
        return params
        
    except Exception as e:
        print(f"⚠️  Error reading bootstrap data for {patient_id}: {e}")
        print(f"⚠️  Falling back to default parameter bounds")
        return create_fitting_parameters(initial_volume, noise_level, no_volume)
