# import
import pandas as pd
import copy
import numpy as np
from smoothData import gaussian, savgol, detrend
from fileSaveHDF5 import saveto_h5_4Dmatrix, saveto_h5_3Dmatrix
from interpData import (RDP_interpolate, interpolate_data, segment_interpolation, remove_zero_padding, 
                        split_and_pad_dataframe, pad_dataframe, pad_data_zeros, split_by_discontinuous_index, 
                        split_into_k_parts)
from normalizeData import (normalize, normalize_EPC, phase_standardize, phase_scale,
                           standardize_robust)
from transformData import (unwrap, phase_log_transform, phase_quantile_transform, phase_robust_scaler,
                           phase_power_transform, phase_log_transform_shift)
from augmentData import add_constant_offset, sub_constant_offset, add_gaussian_noise, add_offset_and_noise

# this gets rid of some random warning when separating data into EPC1 & EPC2
pd.options.mode.chained_assignment = None  # default='warn'

# this function formats the original data exported from the ItemTest program
# made by IMPINJ. The output of this function returns a dataframe list
# which contains RSSI & phase data based on EPC and iteration for 1 gesture
def format(inputs, h5_name, labels, RSSI_val, phase_val, flags, lengths):


    ############################### FUNCTION VARIABLES ###############################
    # init empty lists
    data = []
    EPC_sep = []
    EPC_timestamps = []
    EPC_padded = []
    EPC_interp = []

    # store list variables into readable names
    h5_flag = flags[0]
    norm_flag = flags[1]
    aug_flag = flags[2]

    interp_length = lengths[0]
    threshold = lengths[1]

    #init dictionary to store EPC values
    mapping = {'A10000000000000000000000': 1,
               'A20000000000000000000000': 2,
               'A30000000000000000000000': 3,
               'A40000000000000000000000': 4,
               'A60000000000000000000000': 5,
               'A70000000000000000000000': 6,
               'A80000000000000000000000': 7,
               'A90000000000000000000000': 8}
    
    EPC_count = len(mapping)
    
    ############################### DATA PREFORMATTING ###############################
    # load csv file into dataframe and change header row (deletes everything before)
    for input in inputs:
        print(f'Trying to read csv file: {input}')  # Log the file being processed
        try:
            data.append(pd.read_csv(input, header=2))
        except Exception as e:
            print(f"Error reading file: {input}")
            raise e

      
    # format column names and datatypes for easier preprocessing
    data = [format_datatypes_columns(df, mapping, inputs[i]) for i, df in enumerate(data)]

    # iterate through the dataframe list to split by EPC and get timestamp column
    # must split by EPC first so we can unwrap
    for df in data:
        for j in range(1, EPC_count + 1):
            EPC_timestamps.append(np.sort(df['TimeValue'].values))
            EPC_sep.append(df[df['EPC'] == j])

    if(norm_flag == 1):   
        print(f'---------------------- TRAINING DATA --------------------')
        print(f'Preprocessing {len(data)} gestures')                       
        print(f'Tags            : {EPC_count}')
    else:
        print(f'---------------------- TESTING DATA ---------------------')
        print(f'Preprocessing {len(data)} gestures')                            
        print(f'Tags            : {EPC_count}')

    #for i, df in enumerate(EPC_sep):
    #    print(f'EPC data set {i+1}:')
    #    print(df, '\n')
    
    #################################### NORMALIZE DATA ####################################
    # normalize RSSI based on preferences
    if(norm_flag == 1):
        # RSSI normalization options
        EPC_sep, RSSI_val = normalize(EPC_sep, 'RSSI', norm_flag, RSSI_val)
        #EPC_sep, RSSI_val = normalize_EPC(EPC_sep, EPC_count, 'RSSI', norm_flag, RSSI_val)

        EPC_sep = unwrap(EPC_sep)

        # phase transformation options 
        #EPC_sep, phase_val_tx = phase_robust_scaler(EPC_sep, norm_flag, phase_val_tx)
        #EPC_sep, phase_val_tx = phase_power_transform(EPC_sep, norm_flag, phase_val_tx)

        # phase normalization options
        EPC_sep, phase_val = standardize_robust(EPC_sep, norm_flag, phase_val, 'PhaseAngle')

        #print(f'RSSI values: {RSSI_val}')
        #print(f'Phase values: {phase_val}')

    else:
        # RSSI normalization options
        EPC_sep, RSSI_val = normalize(EPC_sep, 'RSSI', norm_flag, RSSI_val)
        #EPC_sep, RSSI_val = normalize_EPC(EPC_sep, EPC_count, 'RSSI', norm_flag, RSSI_val)

        EPC_sep = unwrap(EPC_sep)

        # phase transformation options
        #EPC_sep, phase_val_tx = phase_robust_scaler(EPC_sep, norm_flag, phase_val_tx)
        #EPC_sep, phase_val_tx = phase_power_transform(EPC_sep, norm_flag, phase_val_tx)

        # phase normalization options
        EPC_sep, phase_val = standardize_robust(EPC_sep, norm_flag, phase_val, 'PhaseAngle')
    
    ##################################### SMOOTH DATA ######################################
    window_length = 5
    polyorder = 2
    #EPC_sep = savgol(EPC_sep, 'RSSI', window_length, polyorder)
    EPC_sep = savgol(EPC_sep, 'PhaseAngle', window_length, polyorder)

    stdv = 1
    #EPC_sep = gaussian(EPC_sep, 'RSSI', stdv)
    EPC_sep = gaussian(EPC_sep, 'PhaseAngle', stdv)

    #for df in EPC_sep
    #    print(f'EPC data set preprocessed {i+1}:')
    #    print(EPC_sep[i], '\n')

    ############################### PAD AND INTERPOLATE DATA ################################
    # pad all EPC separated dataframes with zeros during time reads of entire gesture
    for i, df in enumerate(EPC_sep):
        EPC_padded.append(pad_dataframe(df, EPC_timestamps[i], i, EPC_count))

    # define interpolation type (linear, cubic)
    method = 'linear'
    print(f'Interpolation   : {method} with {interp_length} points')
    print(f'Threshold       : Removed zero padded segments over length {threshold}')

    for i, df in enumerate(EPC_padded):
        interpolated_df = interp_functions(df, i, threshold, interp_length, method)
        EPC_interp.append(interpolated_df)
    
        if (i + 1) % 10000 == 0:
            print(f'Interpolation on dataframe {i + 1}/{len(EPC_padded)}...')

        #if(len(df) != length):
        #    print(f'Dataframe has length {len(df)}')
        #    print(df, '\n')
    
    #################################### AUGMENT EPC DATA ###################################
    if(norm_flag == 1 & aug_flag == 1):
        # create augmented list and add in augmented dataframes
        aug_EPC = copy.deepcopy(EPC_interp)
        print(len(aug_EPC))
        for i, df in enumerate(aug_EPC):
            aug_EPC[i] = add_gaussian_noise(df, RSSI_std=0.015, phase_std=0.15)

            if (i + 1) % 10000 == 0:
                print(f'Augmentation on dataframe {i + 1}/{len(EPC_padded)}...')

        # concatenate all dataframes and labels togther
        EPC_interp = EPC_interp + aug_EPC
        labels = labels + labels

    # use data to save as csv which will return and save unchanged data to h5
    if h5_flag == 1:
        saveto_h5_4Dmatrix(EPC_interp, h5_name, labels, EPC_count, norm_flag) # 4D matrix
        #saveto_h5_3Dmatrix(EPC_interp, h5_name, labels, EPC_count, norm_flag) # 3D matrix

    if (norm_flag == 1):
        return RSSI_val, phase_val
    
def format_datatypes_columns(df, mapping, file_name):
        print(f'Processing file: {file_name}')
    # change which columns are needed and remove spaces
        df = df.iloc[0:, [0,1,4,7]]
        df.columns = df.columns.str.strip()

        # remove A5 tag from all gestures
        df = df[df['EPC'] != 'A50000000000000000000000']

        # rename timestamp column
        df.rename(columns={'// Timestamp' : 'Timestamp'}, inplace = True)

        # parse timestamp data and create new column to have comparable time values
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        try:
            df['Timestamp'] = (df['Timestamp'] - df['Timestamp'].iloc[0]).dt.total_seconds()
        except Exception as e:
            print(f"Error processing file: {file_name}")
            raise e
        df.rename(columns={'Timestamp': 'TimeValue'}, inplace=True)

        # correctly change RSSI column to numbers
        if(df['RSSI'].dtype != 'int64'):
            # replace commas, negative signs, then change to float64s
            df['RSSI'] = df['RSSI'].str.replace(',' , '.', regex = False)
            df['RSSI'] = df['RSSI'].str.replace(r'[^\d.-]', '-', regex = True)
            df['RSSI'] = pd.to_numeric(df['RSSI'], errors='coerce')

        # replace commas and change to float64s
        df['PhaseAngle'] = df['PhaseAngle'].str.replace(',' , '.', regex = False)
        df['PhaseAngle'] = pd.to_numeric(df['PhaseAngle'])

        # replace long EPC values with integers from mapping dictionary
        df['EPC'] = df['EPC'].replace(mapping)

        # call cleaning function to remove duplicate reads
        clean(df)

        return df

def interp_functions(df, i, threshold, interp_length, method):
    #print(f'this is the new zero padded EPC dataframe {i+1}')
    #print(df, '\n')

    df = remove_zero_padding(df, threshold)
    df.reset_index(drop = True, inplace = True)
    #print(f'Here is the dataframe {i+1} after dropping zeros of length < {threshold}')
    #print(df, '\n')

    df = segment_interpolation(df, interp_length, method)
    df.reset_index(drop = True, inplace = True)
    #print(f'Here is the dataframe {i+1} after interpolating to {length}')
    #print(df, '\n')

    return df

def clean(df):
    # Drop duplicate rows based on the 'TimeValue' column, keeping the first occurrence
    df_cleaned = df.drop_duplicates(subset='TimeValue', keep='first')
    
    # Optionally, reset the index if needed
    df_cleaned.reset_index(drop=True, inplace=True)
    
    return df_cleaned