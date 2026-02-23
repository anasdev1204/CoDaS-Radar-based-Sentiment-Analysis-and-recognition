# import
import numpy as np

def add_constant_offset(df, RSSI_offset = 0.05, phase_offset = 1):
    # create a zero mask to find indexs
    zero_mask = (df['RSSI'] == 0) & (df['PhaseAngle'] == 0)

    # add offsets to RSSI and Phase where the data points arent padded zeros
    df.loc[~zero_mask, 'RSSI'] += RSSI_offset
    df.loc[~zero_mask, 'PhaseAngle'] += phase_offset
    
    return df

def sub_constant_offset(df, RSSI_offset = 0.05, phase_offset = 1):
    # create a zero mask to find indexs
    zero_mask = (df['RSSI'] == 0) & (df['PhaseAngle'] == 0)

    # add offsets to RSSI and Phase where the data points arent padded zeros
    df.loc[~zero_mask, 'RSSI'] -= RSSI_offset
    df.loc[~zero_mask, 'PhaseAngle'] -= phase_offset
    
    return df

def add_gaussian_noise(df, RSSI_std = 0.015, phase_std = 0.15):
    # create a zero mask to find indexs
    zero_mask = (df['RSSI'] == 0) & (df['PhaseAngle'] == 0)

    # Generate noise only for the valid (non-zero-masked) rows
    noise_rssi = np.random.normal(0, RSSI_std, size=df.shape[0])
    noise_phase = np.random.normal(0, phase_std, size=df.shape[0])
    
    # Apply the noise where the zero mask is False
    df.loc[~zero_mask, 'RSSI'] += noise_rssi[~zero_mask]
    df.loc[~zero_mask, 'PhaseAngle'] += noise_phase[~zero_mask]
    
    return df

def add_offset_and_noise(df, RSSI_offset = 0.05, phase_offset = 1, RSSI_std = 0.1, phase_std = 0.5):
    # create a zero mask to find indexs
    zero_mask = (df['RSSI'] == 0) & (df['PhaseAngle'] == 0)

    # add offsets to RSSI and Phase where the data points arent padded zeros
    df.loc[~zero_mask, 'RSSI'] += RSSI_offset
    df.loc[~zero_mask, 'PhaseAngle'] += phase_offset

    # Generate noise only for the valid (non-zero-masked) rows
    noise_rssi = np.random.normal(0, RSSI_std, size=df.shape[0])
    noise_phase = np.random.normal(0, phase_std, size=df.shape[0])
    
    # Apply the noise where the zero mask is False
    df.loc[~zero_mask, 'RSSI'] += noise_rssi[~zero_mask]
    df.loc[~zero_mask, 'PhaseAngle'] += noise_phase[~zero_mask]

    return df
    
def sub_offset_and_noise(df, RSSI_offset = 0.05, phase_offset = 1, RSSI_std = 0.1, phase_std = 0.5):
    # create a zero mask to find indexs
    zero_mask = (df['RSSI'] == 0) & (df['PhaseAngle'] == 0)

    # add offsets to RSSI and Phase where the data points arent padded zeros
    df.loc[~zero_mask, 'RSSI'] -= RSSI_offset
    df.loc[~zero_mask, 'PhaseAngle'] -= phase_offset

    # Generate noise only for the valid (non-zero-masked) rows
    noise_rssi = np.random.normal(0, RSSI_std, size=df.shape[0])
    noise_phase = np.random.normal(0, phase_std, size=df.shape[0])
    
    # Apply the noise where the zero mask is False
    df.loc[~zero_mask, 'RSSI'] += noise_rssi[~zero_mask]
    df.loc[~zero_mask, 'PhaseAngle'] += noise_phase[~zero_mask]

    return df

# this function provides a scaling technique to RSSI and phase data.
# SET THE SEED TO NONE BEFORE THE FUNCTION CALL
def scale(df, rssi_scale = (0.9, 1.1), phase_scale = (0.9, 1.1)):
    # create a zero mask to find indexs
    zero_mask = (df['RSSI'] == 0) & (df['PhaseAngle'] == 0)
    
    # calculate a random number in the scale range
    rssi_factor = np.random.uniform(rssi_scale[0], rssi_scale[1])
    phase_factor = np.random.uniform(phase_scale[0], phase_scale[1])

    # scale the RSSI and phase columns appropriately
    df.loc[~zero_mask, 'RSSI'] *= rssi_factor
    df.loc[~zero_mask, 'PhaseAngle'] *= phase_factor

    return df

# this function shifted data columns by a random integer within the range
# SET THE SEED TO NONE BEFORE THE FUNCTION CALL
def shfit_time(df, shift_range = (-5, 5)):

    # calculate a random integer for shift value
    shift_value = np.random.randint(shift_range[0], shift_range[1])

    # apply shift to both columns
    df.iloc[:, 2:] = df.iloc[:, 2:].shift(shift_value, fill_value = 0)

    return df