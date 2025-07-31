##############################
## Configuration & Logging
##############################

#Percentile for plots
P_PERCENTILE = 5.0 # 2.5

# Patient configuration with train/test split settings
PATIENT_CONFIG = {
    'PID_001': {'mode': 'last_n_points', 'n_points': 1, 'valid': True},
    'PID_002': {'valid': False},
    'PID_003': {'mode': 'last_n_points', 'n_points': 1, 'valid': True},
    'PID_004': {'mode': 'remove_last_predict', 'remove_last': 1, 'predict_remaining': 1, 'valid': True},
    'PID_005': {'mode': 'remove_last_predict', 'remove_last': 1, 'predict_remaining': 1, 'valid': True},
    'PID_006': {'mode': 'remove_last_predict', 'remove_last': 1, 'predict_remaining': 1, 'valid': True},
    'PID_007': {'valid': False},
    'PID_008': {'mode': 'remove_last_predict', 'remove_last': 1, 'predict_remaining': 1, 'valid': True},
    'PID_009': {'mode': 'pre_rt_points', 'valid': True},
    'PID_010': {'mode': 'remove_last_predict', 'remove_last': 2, 'predict_remaining': 1, 'valid': True},
    'PID_011': {'mode': 'last_n_points', 'n_points': 1, 'valid': True},
    'PID_012': {'mode': 'remove_last_predict', 'remove_last': 2, 'predict_remaining': 1, 'valid': True},
    'PID_013': {'mode': 'last_n_points', 'n_points': 1, 'valid': True},
}
