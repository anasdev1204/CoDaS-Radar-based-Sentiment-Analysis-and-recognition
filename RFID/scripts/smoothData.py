# import
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter, detrend
import statsmodels.api as sm

# rolling mean filter: smooths data by taking a moving average of each point
# window is the size of how many adjacent points are included
# ***** this function deletes window - 1 points from the beginning of a dataframe *****
def moving_avg(EPC_sep, col, wlen = 3):
    count = 0

    for df in EPC_sep:
        if not df.empty:
            df[col] = df[col].rolling(window = wlen).mean()
        else:
            count += 1

    print(f'Moving average filter applied to {col}, passed on {count} dataframes')
    return EPC_sep

# Gaussian filter: smooths data by applying a Gaussian kernel which gives more weight to the central 
# point and less weight as you traverse out
# sigma is the stdv of the kernel, larger sigma, more smoothing
# large stdvs shrink data over smooth sharp changes
def gaussian(EPC_sep, col, stdv = 0.5):
    # low stdv: 0.5 to 2
    # mid stdv: 2 to 5
    # high stdv: > 5

    count = 0

    for df in EPC_sep:
        if not df.empty:
            df[col] = gaussian_filter1d(df[col], sigma = stdv)
        else:
            count += 1

    print(f'Gaussian Filter : {col} smoothed, skipped {count} dataframes')
    return EPC_sep

# savgol filter: smooths data by using successive subsets of adjacent datapoints with a low-degree
# polynomial by the method of linear least squares
# window_length must be odd, less than or equal to amount of data values, and greater than polyorder
def savgol(EPC_sep, col, wlen = 5, polyorder = 2):
    count = 0
    
    for i in range(len(EPC_sep)):
        # KEEP THE GREATER THAN, DO NOT CHANGE TO GREATER OR EQUAL THAN
        if (EPC_sep[i].shape[0] > wlen):
            EPC_sep[i][col] = savgol_filter(EPC_sep[i][col], window_length = wlen, polyorder = polyorder)
        else:
            count += 1

    print(f'Savgol Filter   : {col} smoothed, skipped {count} dataframes')
    return EPC_sep

# Detrend function: removes any trend components of data
def detrend(EPC_sep, col):
    count = 0

    for df in EPC_sep:
        if not df.empty:
            df[col] = detrend(df[col])
        else:
            count += 1

    print(f'Detrend filter applied to {col}, passed on {count} dataframes')
    return EPC_sep

# applies exponentially decreasing weights to data points, 
# which smooths data while giving more importance to recent observations
def exponential_moving_average(EPC_sep, col, a = 0.1):
    count = 0
    for df in EPC_sep:
        if not df.empty:
            df[col] = df[col].ewm(alpha = a, adjust=False).mean()
        else:
            count += 1

    print(f'EMA filter applied to {col}, passed on {count} dataframes')
    return EPC_sep

# fits local regressions to smooth data, adapting to varying trends and providing a flexible fit
# locally weighted scatter plot
def lowess_smooth(EPC_sep, col, frac = 0.1):
    count = 0

    for df in EPC_sep:
        if not df.empty:
            lowess = sm.nonparametric.lowess(df[col], df.index, frac = frac)
            df[col] = lowess[:, 1]
            df[col] = lowess[:, 1]
        else:
            count+= 1
    print(f'LOWESS filter applied to {col}, passed on {count} dataframes')
    return EPC_sep