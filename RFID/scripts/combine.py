# import
import os
from fileManage import get_gesture_folders
import shutil

def copy_files(gesture_directory, destination_directory):
    # walks through directory tree to pull out csv file paths
    for root, _, files in os.walk(gesture_directory):
        for file in files:
            if file.endswith('.csv'):
                src_file_path = os.path.join(root, file)

                # copy the file to the destination directory
                shutil.copy2(src_file_path, destination_directory)

def combine_participants(participant_directories, combined_directory):
    print('---------------- COMBINING TRAINING DATA ----------------')
    all_gesture_paths = []
    participant_count = len(participant_directories)

    combined_gesture_directories = get_gesture_folders(combined_directory)
    
    for i in range(participant_count):
        all_gesture_paths.append(get_gesture_folders(participant_directories[i]))

    for i in range(participant_count):
        for j in range(len(all_gesture_paths[0])):
            copy_files(all_gesture_paths[i][j], combined_gesture_directories[j])
        #print(f'Just copied {len(all_gesture_paths[0])} gestures for participant {i+1}')
    
    print(f'Combined {len(participant_directories)} participants\n')
        
def get_participants():
    participant_directories = [r'\SS 1.5m',
                               r'\YL 1.5m',
                               r'\MS 1.5m',
                               r'\SH 1.5m',
                               r'\LJ 1.5m'
                               ]
    
    return participant_directories