# import
import os
import glob
import re

# this function clears a specific file folder
def clear_folder(folder_path, ext):
    # Construct the file pattern to match extension in specific folder
    file_pattern = os.path.join(folder_path, ext)
    
    # List all files in the folder
    delete_files = glob.glob(file_pattern)
    
    # Iterate through the list of CSV files and delete each file
    for file in delete_files:
        try:
            os.remove(file)
            #print(f'Deleted: {file}')
        except OSError as e:
            print(f"Error: {file} - {e.strerror}")

def clear_directory(directory_path, ext):
    print('\n-------------- DELETING COMBINED DIRECTORY --------------')
    for gesture_folder in sorted(os.listdir(directory_path)):
        gesture_path = os.path.join(directory_path, gesture_folder)
        clear_folder(gesture_path, ext)

    print(f'Deleted all {ext} file in {directory_path}\n')

# this function finds all csv files inside a specified folder from main to
# store the correct full path of each dataset in an array
def get_csv_filenames(folder_path):       
    csv_path = []
    labels = []

    last_folder = os.path.basename(folder_path)
    number = extract_number_from_string(last_folder)

    for item in os.listdir(folder_path):
        if item == '.DS_Store':
            file_path = os.path.join(folder_path, item)
            os.remove(file_path)
            #print(f'Removed {file_path}')

    for csv_file in os.listdir(folder_path):
        if csv_file.endswith('.csv'):
            csv_path.append(os.path.join(folder_path, csv_file))
            labels.append(f'gesture{number}')
            
    # return file names list
    return csv_path, labels

# this function creates a list with all the csv files in a directory
def get_csv_all(root_dir):
    # init list to hold all csv_path names
    csv_path = []
    labels = []

    for item in os.listdir(root_dir):
        if item == '.DS_Store':
            file_path = os.path.join(root_dir, item)
            os.remove(file_path)
            #print(f'Removed {file_path}')

    # Read each subfolder (gesture type)
    for gesture_folder in sorted(os.listdir(root_dir), key = numerical_sort):
        # ignore gesture folders
        #if gesture_folder in ['18 Two-hand Inward Circles']:
        #    continue

        # save gesture path per participant
        gesture_path = os.path.join(root_dir, gesture_folder)

        number = extract_number_from_string(gesture_folder)

        # Read each CSV file in the gesture subfolder and create label
        for csv_file in os.listdir(gesture_path):
            if csv_file.endswith('.csv'):
                csv_path.append(os.path.join(gesture_path, csv_file))
                name = f'gesture{number}'
                labels.append(name)

    return csv_path, labels

# this function is to return a list of directory paths of each gesture per 1 participant
# sorted numerically
def get_gesture_folders(participant_directory):
    # create empty list for gesture paths
    gesture_path = []

    for item in os.listdir(participant_directory):
        if item == '.DS_Store':
            file_path = os.path.join(participant_directory, item)
            os.remove(file_path)

    # find all the gesture folder per participant and sort them numerically
    for gesture_folder in sorted(os.listdir(participant_directory), key = numerical_sort):
        # ignore gesture folders
        #if gesture_folder in ['18 Two-hand Inward Circles']:
        #    continue

        # this is a list of all the path names to each gesture per 1 participant
        gesture_path.append(os.path.join(participant_directory, gesture_folder))
        #print(gesture_path[-1])

    return gesture_path

# the function extracts the numerical part from a string for sorting
def numerical_sort(value):
    # uses a regular expression to find all sequences of digits in the input string value
    # r'\d+'    : raw string containing the regular exp pattern (can be anywhere in string)
    # \d        : any digit
    # +         : will add on any preceding numerical values
    parts = re.findall(r'\d+', value) 

    # finds the first element in the parts array and converts to integer
    # if parts is empty return 0
    return int(parts[0]) if parts else 0

def extract_number_from_string(s):
    match = re.match(r'^(\d+)', s)
    if match:
        return int(match.group(1))
    return None