# import
import numpy as np
from sklearn.preprocessing import QuantileTransformer, RobustScaler, PowerTransformer
from scipy import stats

# unwrap all phase data in [0, 2pi] by EPC per gesture 
def unwrap(EPC_sep):
    for df in EPC_sep:
        if not df.empty:
            df['PhaseAngle'] = np.unwrap(df['PhaseAngle'])

    return EPC_sep
# this function returns a log scale of the phase data to compress the data before normalization
# The transformed values are not strictly bounded or scaled, which might lead to inconsistencies 
# when used with subsequent normalization techniques
def phase_log_transform(EPC_sep):
    for i in range(len(EPC_sep)):
        if not EPC_sep[i].empty:
            # Apply log transformation (adding a small constant to avoid log(0))
            EPC_sep[i]['PhaseAngle'] = np.log1p(np.abs(EPC_sep[i]['PhaseAngle'])) * np.sign(EPC_sep[i]['PhaseAngle'])

    return EPC_sep

# The transformation may not fully handle negative values or outliers in the same way, 
# which could impact how well it integrates with other normalization techniques.
def phase_log_transform_shift(EPC_sep, phase_flag, shift_val):
    if phase_flag == 1:
        shift_val = None

        # Collect all phase data into a single array and compute shift_val
        all_phases = np.concatenate([df['PhaseAngle'].values for df in EPC_sep if not df.empty])
        shift_val = -all_phases.min() + 1 
    
    # Apply transformation
    for df in EPC_sep:
        if not df.empty:
            df['PhaseAngle'] = np.log(df['PhaseAngle'] + shift_val)
            df.reset_index(drop=True, inplace=True)
    
    return EPC_sep, shift_val

# Maps the data to a uniform or normal distribution using quantiles. 
# This method can handle non-linear relationships and skewed distributions
def phase_quantile_transform(EPC_sep, phase_flag, transformer, dist = 'uniform'):
    # Check if phase normalization should be applied
    if phase_flag == 1:

        transformer = None

        # Combine phase data from all dataframes into a single array
        phase_data = np.concatenate([df['PhaseAngle'].values for df in EPC_sep if not df.empty])
        
        # Initialize the QuantileTransformer, which will map the data to a distribution
        # uniform: Spreads data evenly across the range [0, 1], which might be useful in certain scenarios
        # normal: Converts data to follow a normal distribution, which can be beneficial for models that assume normality
        transformer = QuantileTransformer(output_distribution = dist)
        
        # Fit the transformer on the entire phase data (this computes the quantiles)
        transformer.fit(phase_data.reshape(-1, 1))
    
    # Apply the fitted transformer to each dataframe
    for i in range(len(EPC_sep)):
        if not EPC_sep[i].empty:
            # Transform the phase data for each dataframe
            EPC_sep[i]['PhaseAngle'] = transformer.transform(EPC_sep[i]['PhaseAngle'].values.reshape(-1, 1)).flatten()

    return EPC_sep, transformer

# Scales features using statistics that are robust to outliers, specifically using the median and the interquartile range (IQR). 
# This technique is particularly useful if your phase data has outliers or non-standard distributions
def phase_robust_scaler(EPC_sep, phase_flag, scaler):
    if phase_flag == 1:

        scaler = None

        scaler = RobustScaler()
        
        # Collect phase data from all EPCs
        all_phases = np.concatenate([df['PhaseAngle'].values for df in EPC_sep if not df.empty])
        
        # Fit the scaler
        scaler.fit(all_phases.reshape(-1, 1))
    
    # Apply transformation
    for df in EPC_sep:
        if not df.empty:
            df['PhaseAngle'] = scaler.transform(df['PhaseAngle'].values.reshape(-1, 1)).flatten()
            df.reset_index(drop=True, inplace=True)
    
    print('Robust Transformation on PhaseAngle')
    return EPC_sep, scaler

# Applies a power transformation to make the data more Gaussian-like. 
# It can help stabilize variance and make the data more normalized.
def phase_power_transform(EPC_sep, phase_flag, transformer):
    if phase_flag == 1:
        transformer = None

        transformer = PowerTransformer(method='yeo-johnson')
        
        # Collect phase data from all EPCs
        all_phases = np.concatenate([df['PhaseAngle'].values for df in EPC_sep if not df.empty])
        
        # Fit the transformer
        transformer.fit(all_phases.reshape(-1, 1))
    
    # Apply transformation
    for df in EPC_sep:
        if not df.empty:
            df['PhaseAngle'] = transformer.transform(df['PhaseAngle'].values.reshape(-1, 1)).flatten()
            df.reset_index(drop=True, inplace=True)
    
    print('Power Transformation on PhaseAngle')
    return EPC_sep, transformer