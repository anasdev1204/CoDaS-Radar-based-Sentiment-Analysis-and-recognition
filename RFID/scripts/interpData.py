#import 
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d, CubicSpline
# from rdp import rdp
from simplification.cutil import simplify_coords

# this function simplifies dataframe by using the Ramer-Douglas-Peucker algorithm and then interpolating
# the data to the fixed length. If the length is smaller than the target length, the function sends the 
# data to be interpolated
def RDP_interpolate(data, target_length, method, ep = 0.5):
    # convert to numpy array for processing
    data_array = data.values.astype(float)
    print(data_array.shape)

    if len(data) > target_length:
        # apply RDP to reduce points while preserving shape
        # lower the epsilon, more preservation of data
        # convert to dataframe
        simplified_data = simplify_coords(data_array, epsilon = ep)
        simplified_df = pd.DataFrame(simplified_data, columns = data.columns)

        # check if simplified data is too small
        if len(simplified_data) < 2:
            raise ValueError("RDP simplification resulted in too few points to perform interpolation.")

        # interpolate the simplified dataframe to the target length
        interpolated_data =  interpolate_data(simplified_df, target_length, method)
    
    else:
        # interpolate data if length is smaller than target
        interpolated_data = interpolate_data(data, target_length, method)

    # convert and return as dataframe
    interpolated_df = pd.DataFrame(interpolated_data, columns = data.columns)
    return interpolated_df

# this function interpolates data to a specific length
def interpolate_data(data, target_length, method):
    # store length in variable
    original_length = len(data)

    # return data if it doesnt need to be interpolated
    if original_length == target_length:
        return data
    
    # evenly spaced arrays of numbers over a specified interval
    x_original = np.linspace(0, 1, original_length)
    x_target = np.linspace(0, 1, target_length)

    # empty array of target_length x data columns (shape[1])
    interpolated_data = np.zeros((target_length, data.shape[1]))

    # ensure data is only numeric
    data_array = data.values.astype(float)

    # iterate over the each column to linear interpolate in rows near the indices of the evenly spaced array values
    for i in range(data.shape[1]):
        # convert to numpy array for processing
        column_data = data_array[:, i]

        if len(x_original) < 2 or len(column_data) < 2:
            EPC = data['EPC'].iloc[0]
            data = pad_data_zeros(data, target_length, -1, EPC)
            return data

        # interpolate and return
        if method == 'linear':
            f = interp1d(x_original, column_data, kind = 'linear', fill_value = "extrapolate")
        elif method == 'cubic':
            f = CubicSpline(x_original, column_data, extrapolate = True)
        else:
            raise ValueError(f"Interpolation method '{method}' is not supported.")
        
        # apply the interpolation function to the target x values
        interpolated_data[:, i] = f(x_target)
        
    # convert and return as dataframe
    interpolated_df = pd.DataFrame(interpolated_data, columns = data.columns)
    interpolated_df.index = np.linspace(data.index.min(), data.index.max(), target_length)
    return interpolated_df

# this function splits each padded dataframe into multiple dataframes that contain zeros or valid data.
# dependent of the condition met by the total length of either the zeros or valid data, there is a 
# certain interpolation technique
def segment_interpolation(df, interp_length, method):
    # Identify columns to check for zero padding and create a zero mask 
    columns_to_check = ['RSSI', 'PhaseAngle']
    zero_mask = (df[columns_to_check] == 0).all(axis=1)
    
    ################################## SPLIT ZEROS ##################################
    # Split passed dataframe into zeros and valid data, find length
    df_zero = df[zero_mask]
    df_data = df[~zero_mask]
    zero_length = len(df_zero)
    data_length = len(df_data)

    # create list of segmented dataframes
    zero_segmented_dfs = split_by_discontinuous_index(df_zero)
    data_segmented_dfs = split_by_discontinuous_index(df_data)

    ############################### CASE 1 & 2: no zeros or data ###############################
    if (zero_length == 0 or data_length == 0):
        df = interpolate_data(df, interp_length, method)
        return df
    
    ############################### CASE 3: perfect length ###############################
    elif ((zero_length + data_length) == interp_length):
        return df
    
    ############################### CASE 4 & 5: interpolate zeros only ###############################
    elif ((zero_length + data_length) > interp_length):
        if(data_length <= interp_length):
            # create dataframe list that will hold dfs to concatenate
            dfs_concat = [df_data]

            # find all weighted interp lengths dependent on how many segmented zero dataframes there are
            interp_split_nums = split_into_k_parts(interp_length - data_length, len(zero_segmented_dfs))
            #interp_split_nums = proportional_split(zero_segmented_dfs, interp_length - data_length)

            # iterate through all zero segmented dataframes (could still be a list of 1)
            for i, df_zeros in enumerate(zero_segmented_dfs):
                # append the interpolated zero segemtned dataframes to dataframe list
                interpolated_zeros = interpolate_data(df_zeros, interp_split_nums[i], method)
                dfs_concat.append(interpolated_zeros)

            # concatenate all dataframes together (OG data and all segmented zeros)
            df_interp = pd.concat(dfs_concat, ignore_index=False)
        
        else:
            df_interp = interpolate_data(df_data, interp_length, method).sort_index()
        
        return df_interp.sort_index()
    ############################### CASE 6: interpolate data only ###############################
    elif ((zero_length + data_length) < interp_length):
        # create dataframe list that will hold dfs to concatenate
        dfs_concat = [df_zero]

        # find all weighted interp lengths dependent on how many segmented data dataframes there are
        interp_split_nums = split_into_k_parts(interp_length - zero_length, len(data_segmented_dfs))
        #interp_split_nums = proportional_split(data_segmented_dfs, interp_length - zero_length)

        # iterate through all data segmented dataframes (could still be a list of 1)
        for i, df_data in enumerate(data_segmented_dfs):
            # append the interpolated zero segemtned dataframes to dataframe list
            interpolated_data = interpolate_data(df_data, interp_split_nums[i], method)
            dfs_concat.append(interpolated_data)
            
        # concatenate all dataframes together (OG data and all segmented zeros)
        df_interp = pd.concat(dfs_concat, ignore_index=False)
        return df_interp.sort_index()

    print('\nUNEXPECTED CASE:')
    print(f'Data length: {data_length}, Zero length: {zero_length}')
    #print(df, '\n')
    return df

# this function removes section of zeros that are less than the threshold length
def remove_zero_padding(df, threshold):
    # Identify columns to check for zero padding and create df copy to edit
    columns_to_check = ['RSSI', 'PhaseAngle']
    
    # Create a mask where zeros are present and find continuous zero segements
    zero_mask = (df[columns_to_check] == 0).all(axis=1)
    df['ZeroSegment'] = (zero_mask != zero_mask.shift()).cumsum()
    
    # size the continuous zeros and index segments are under the threshold
    segment_stats = df.groupby('ZeroSegment').size()
    segments_to_remove = segment_stats[segment_stats <= threshold].index
    
    ## Replace zeros with NaN
    ## replace only if the value is zero and it is in a segment that qualifies for replacement
    for segment in segments_to_remove:
        segment_mask = (df['ZeroSegment'] == segment)
        df.loc[segment_mask & zero_mask, columns_to_check] = np.nan
    
    # Drop rows with NaN values and helper column
    df = df.dropna()
    df.drop(columns=['ZeroSegment'], inplace=True)
    
    return df

# this function splits the padded data for a complete gesture, pads EPCs dataframes with 
# zeros to match up with entire gesture time stamp, then send a concatenated dataframe back
def split_and_pad_dataframe(df, EPC_count):
    EPC_sep = []
    split_dfs = []

    # get the full range of timestamps
    full_timestamps = np.sort(df['TimeValue'].values)

    # split the DataFrame by EPC
    for i in range(1, EPC_count + 1):
        EPC_sep.append(df[df['EPC'] == i])
    
    for i, df in enumerate(EPC_sep):
        # Step 3: Reindex with full timestamps
        df = df.drop_duplicates(subset='TimeValue', keep='first')
        df = df.set_index('TimeValue').reindex(full_timestamps, fill_value=0)
        
        # Step 4: Fill missing values with zeros
        df['EPC'] = i + 1
        
        # Reset the index to keep the Timestamp as a column
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'TimeValue'}, inplace=True)
        
        # Store the padded DataFrame
        split_dfs.append(df)

    combined_df = pd.concat(split_dfs, ignore_index=True)

    return combined_df

# this function pads EPC dataframes with zeros to match up with entire gesture time stamp
def pad_dataframe(df, EPC_ts, i, EPC_count):
    # reindex with full timestamps
    df = df.drop_duplicates(subset='TimeValue', keep='first')
    df = df.set_index('TimeValue').reindex(EPC_ts, fill_value=0)
    
    # fill missing EPC values
    i += 1
    EPC = (i % EPC_count)

    if EPC == 0:
        EPC = 8

    if i == -1:
        EPC = EPC_count

    df['EPC'] = EPC
    
    # reset the index to keep the new TimeValue column
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'TimeValue'}, inplace=True)

    return df

# this function pads empty EPC dataframes with zeros
def pad_data_zeros(EPC_sep, max_length, i, EPC_count):
    #print('Padding...')

    padding = np.zeros((max_length - len(EPC_sep), EPC_sep.shape[1]))
    padded_data = np.vstack((EPC_sep, padding))

    padded_dataframe = pd.DataFrame(padded_data, columns = EPC_sep.columns)

    i += 1
    EPC = (i % EPC_count)

    if EPC == 0:
        EPC = 8

    if i == -1:
        EPC = EPC_count

    #print(f'This data frame was empty, i think its from tag {EPC}')

    for j in range(len(padded_dataframe)):
        padded_dataframe['EPC'][j] = EPC

    return padded_dataframe

# this function helps split dataframes into segements where there are discontinuties
def split_by_discontinuous_index(df):
    # Calculate the difference between consecutive indices, and a discontinuity
    index_diff = df.index.to_series().diff()
    discontinuities = index_diff > 1

    # Assign a number to each discont segment to split dataframe by
    segment_ids = discontinuities.cumsum()
    segmented_dfs = [group for _, group in df.groupby(segment_ids)]
    
    return segmented_dfs

# this function can divide a number n into k parts as equally as possible
def split_into_k_parts(n, k):
    # Base size of each part
    base_part = n // k
    
    # Number of parts that need an extra 1 to account for the remainder
    remainder = n % k
    
    # Create the parts list
    parts = [base_part + 1 if i < remainder else base_part for i in range(k)]
   
    return parts

def proportional_split(seg_dfs, desired_total_length):
    df_lens = [len(df) for df in seg_dfs]

    original_total_length = sum(df_lens)
    desired_lengths = [
        round((length / original_total_length) * desired_total_length)
        for length in df_lens
    ]

    # Adjust the lengths if the sum is less than or greater than desired_total_length
    while sum(desired_lengths) != desired_total_length:
        if sum(desired_lengths) < desired_total_length:
            # Add 1 to the max length
            max_idx = desired_lengths.index(max(desired_lengths))
            desired_lengths[max_idx] += 1

        elif sum(desired_lengths) > desired_total_length:
            # Subtract 1 from the min length
            min_idx = desired_lengths.index(min(desired_lengths))
            desired_lengths[min_idx] -= 1

    return desired_lengths
