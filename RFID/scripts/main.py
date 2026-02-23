# import functions
from formatData import format
from fileManage import get_csv_all, clear_folder, clear_directory, get_csv_filenames
from combine import combine_participants, get_participants
import sys
import numpy as np
np.random.seed(42)

# this function is to reformat and save a .h5 file for set of gestures
def main(combined_directory, root_dir):

    ############################ SET FLAGS ##########################
    h5_flag = 1
    norm_flag = 1
    aug_flag = 1
    flags = [h5_flag, norm_flag, aug_flag]

    interpolation_length = 30
    threshold = 4
    lengths = [interpolation_length, threshold]

    #################### COMPLETE TRAINING DATA #####################
    if 'Combined' not in combined_directory:
        print('\n*********************************************************************')
        print('* The destination directory is not the allocated combined directory *')
        print('*********************************************************************\n')
        sys.exit(1)

    # clear the combined data directory and HDF5 folder
    clear_directory(r'\Combined', '*.csv')
    clear_folder(r'\HFiveFormat', '*.h5')

    # find all wanted participants to be apart of training data
    # and store in the combined directory
    participant_directories = get_participants()
    combine_participants(participant_directories, combined_directory)
    csv_path_combined, comb_labels = get_csv_all(combined_directory)

    # define h5 file name and data normalization values
    h5_name = 'ply_data_train'
    RSSI_val = []
    phase_val = []
    phase_val_tx = []

    # format data
    RSSI_val, phase_val = format(csv_path_combined, h5_name, comb_labels, RSSI_val, phase_val, flags, lengths)

    ###################### COMPLETE TESTING DATA #####################
    # get all csv files from testing participant directory
    csv_path, labels = get_csv_all(root_dir)

    # define h5 file name and turn off normalization
    h5_name = 'ply_data_test'
    flags[1] = 0 # normalization flag

    # format data
    format(csv_path, h5_name, labels, RSSI_val, phase_val, flags, lengths)

if __name__ == '__main__':
    #declare root dir that contains subfolders for each gesture & set combine flag
    combined_directory = r'\Combined'

    # declare root dir that contains subfolders for each gesture for test participant
    root_dir = r'\SG 1.5m'

    main(combined_directory, root_dir)


