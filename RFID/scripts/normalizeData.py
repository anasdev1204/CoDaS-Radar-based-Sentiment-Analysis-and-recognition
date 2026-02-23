# import
import numpy as np
from scipy.stats import median_abs_deviation

# this function normalizes RSSI data by EPC
def normalize_EPC(EPC_sep, EPC_count, col, data_flag, data_val):
    if data_flag == 1:
        # initialize dictionaries to store min & max RSSI values by EPC 
        # stores giant numbers to be replaced after the first iteration
        min_dict = {i: float('inf') for i in range(1, EPC_count + 1)}
        max_dict = {i: float('-inf') for i in range(1, EPC_count + 1)}
        
        # iterate and find min & max RSSI by EPC
        for df in EPC_sep:
            if not df.empty:
                EPC_value = df['EPC'].iloc[0]
                min_dict[EPC_value] = min(min_dict[EPC_value], df[col].min())
                max_dict[EPC_value] = max(max_dict[EPC_value], df[col].max())
        
        #print(RSSI_min_dict)
        #print(RSSI_max_dict)
        # store min and max values for each EPC
        min_list = [min_dict[i] for i in range(1, EPC_count + 1)]
        max_list = [max_dict[i] for i in range(1, EPC_count + 1)]
        data_val.append(min_list)
        data_val.append(max_list)

    # normalize the RSSI data in each dataframe
    for df in EPC_sep:
        if not df.empty:
            EPC_value = df['EPC'].iloc[0]
            min_val = data_val[0][EPC_value - 1]
            max_val = data_val[1][EPC_value - 1]
            df[col] = (df[col] - min_val) / (max_val - min_val)

    print(f'Min-Max by EPC Normalization on {col}')
    return EPC_sep, data_val

# this function normalizes the RSSI data based on the absolute minimum RSSI value
def normalize(EPC_sep, col, data_flag, data_val):
    # find absolute min of RSSI
    if(data_flag == 1):
        min_list = []
        max_list = []

        for i in range(len(EPC_sep)):
            if (EPC_sep[i].empty == False):
                min_list.append(min(EPC_sep[i][col]))
                max_list.append(max(EPC_sep[i][col]))

        data_val.append(min(min_list))
        data_val.append(max(max_list))
    
    for i in range(len(EPC_sep)):
        # normalize all RSSI data
        EPC_sep[i][col] = (EPC_sep[i][col] - data_val[0]) / (data_val[1] - data_val[0])

    print(f'Normalization   : {col} min-max')
    return EPC_sep, data_val

# this function standardizes the phase values with a mean of 0 and standard deviation of 1
# this can help for data not suited for min-max normalization
def phase_standardize(EPC_sep, phase_flag, phase_val):
    # Calculate mean and standard deviation of unwrapped phase angles
    if phase_flag == 1:
        phase_mean_list = []
        phase_std_list = []
        for i in range(len(EPC_sep)):
            if not EPC_sep[i].empty:
                phase_mean_list.append(np.mean(EPC_sep[i]['PhaseAngle']))
                phase_std_list.append(np.std(EPC_sep[i]['PhaseAngle']))

        # Store the global mean and standard deviation
        phase_val.append(np.mean(phase_mean_list))
        phase_val.append(np.mean(phase_std_list))
    
    # Apply standardization to phase angles
    for i in range(len(EPC_sep)):
        EPC_sep[i]['PhaseAngle'] = (EPC_sep[i]['PhaseAngle'] - phase_val[0]) / phase_val[1]

    return EPC_sep, phase_val

# this function scales the data to a -1 to 1 range (Z-score normalization) instead of a 0 to 1 range (standard normalization)
def phase_scale(EPC_sep, phase_flag, phase_val):
    if phase_flag == 1:
        phase_min_list = []
        phase_max_list = []
        for i in range(len(EPC_sep)):
            if not EPC_sep[i].empty:
                phase_min_list.append(min(EPC_sep[i]['PhaseAngle']))
                phase_max_list.append(max(EPC_sep[i]['PhaseAngle']))

        phase_val.append(min(phase_min_list))
        phase_val.append(max(phase_max_list))
    
    for i in range(len(EPC_sep)):
        # Scale to -1 to 1 range
        EPC_sep[i]['PhaseAngle'] = 2 * (EPC_sep[i]['PhaseAngle'] - phase_val[0]) / (phase_val[1] - phase_val[0]) - 1

    return EPC_sep, phase_val

# Use robust measures of central tendency and dispersion (like median and MAD) to perform Z-score normalization
def standardize_robust(EPC_sep, phase_flag, phase_val, col):
    if phase_flag == 1:
        # Collect phase data from all EPCs
        all_phases = np.concatenate([df[col].values for df in EPC_sep if not df.empty])
        
        # Compute robust statistics
        median = np.median(all_phases)
        mad = median_abs_deviation(all_phases)
        phase_val.append(median)
        phase_val.append(mad)
    
    # Normalize phase data
    for df in EPC_sep:
        if not df.empty:
            df[col] = (df[col] - phase_val[0]) / phase_val[1]
            df.reset_index(drop=True, inplace=True)
    
    print(f'Standardization : {col} robustly')

    return EPC_sep, phase_val