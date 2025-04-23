import scipy.io
import os
import numpy as np
import json

parent_dir = r'C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Data Files\StraightWalking\MatLab\IMU'

IMU_data = {}

try:  # For debugging purposes
    # Looping over subjects
    for subject_folder in os.listdir(parent_dir):
        print(f'Starting conversion for patient {subject_folder}')
        # Creating a sub-dict for each subject
        IMU_data[subject_folder] = {}

        # Creating path to subject folder
        subject_dir = os.path.join(parent_dir, subject_folder)

        # Looping over all trials in the subject folder
        for trial_idx, trial_file in enumerate(os.listdir(subject_dir)):
            # Creating a sub-dict for each trial
            IMU_data[subject_folder][trial_idx] = {}

            # Creating filepath to read the .mat file
            file_path = os.path.join(subject_dir, trial_file)

            data_struct = scipy.io.loadmat(file_path)['restructrawdata']
            data_struct = np.squeeze(data_struct)

            # Looping over the sensors
            for sensor in data_struct:
                sensor_id, _, acc, gyr, mag, time, *other = sensor.item()
                if len(other) > 2:
                    print(f'Data anomaly detected for {file_path} sensor: {sensor_id}')

                # Storing acceleration, gyroscope, magnetometer and time for each sensor
                # (We need to possibly apply rotations to the coordinate systems of these sensors)
                IMU_data[subject_folder][trial_idx][sensor_id.item()] = {
                    'Accelerometer': {'x': acc[0].tolist(),
                                      'y': acc[1].tolist(),
                                      'z': acc[2].tolist()},
                    'Gyroscope':  {'x': gyr[0].tolist(),
                                   'y': gyr[1].tolist(),
                                   'z': gyr[2].tolist()},
                    'Magnetometer': {'x': mag[0].tolist(),
                                     'y': mag[1].tolist(),
                                     'z': mag[2].tolist()}
                }

            # At the end we add the timestamps to the measurements, since these are the same across sensors
            IMU_data[subject_folder][trial_idx]['Time'] = np.squeeze(time).tolist()
except:
    print('Error has occurred')
    breakpoint()

# Saving the dictionary to a JSON file

print('Converted all .mat files, saving....')
with open(r'C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Data Files\StraightWalking\JSON\IMU_raw.json', 'w') as file:
    json.dump(IMU_data, file, indent=2)
print('JSON File successfully saved')

