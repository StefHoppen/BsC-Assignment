import scipy.io
import os
import numpy as np
import json

parent_dir = r'C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Data Files\StraightWalking\MatLab\IMU'

IMU_raw_data = {}
IMU_rot_data = {}


devices = {11798771: ('LeftFoot', 0),
           11798777: ('RightFoot', 1),
           11798809: ('Sternum', 2),
           11798873: ('Pelvis', 3)}

# Looping over subjects
for subject_folder in os.listdir(parent_dir):
    print(f'Starting conversion for patient {subject_folder}')
    # Creating a sub-dict for each subject
    IMU_raw_data[subject_folder] = {}
    IMU_rot_data[subject_folder] = {}

    # Creating path to subject folder
    subject_dir = os.path.join(parent_dir, subject_folder)

    # Looping over all trials in the subject folder
    for trial_idx, trial_file in enumerate(os.listdir(subject_dir)):
        # Creating a sub-dict for each trial
        IMU_raw_data[subject_folder][trial_idx] = {}
        IMU_rot_data[subject_folder][trial_idx] = {}

        # Creating filepath to read the .mat file
        file_path = os.path.join(subject_dir, trial_file)

        # Reading the .mat file
        data_tuple = scipy.io.loadmat(file_path)['rawdata'].item()

        # Extracting components of the .mat file
        time_data, gyr_data, acc_data, mag_data, _, _, _, _, _, rot_data, *other = data_tuple
        device_id = np.squeeze(other[-1])

        # Storing timesteps for all sensors since they are the same
        IMU_raw_data[subject_folder][trial_idx]['Time'] = np.round(np.squeeze(np.squeeze(time_data)[0]), 3).tolist()
        IMU_rot_data[subject_folder][trial_idx]['Time'] = np.round(np.squeeze(np.squeeze(time_data)[0]), 3).tolist()

        # Checking if device_ids(sensors) match
        for idx, id_ in enumerate(device_id):
            id_ = id_.item()
            if not devices[id_][1] == idx:
                raise KeyError('Device ID listed in device dict does not match with the .mat file')

        # Looping over devices
        for id_, (dev_str, idx) in devices.items():
            # Creating subdict for each sensor
            IMU_raw_data[subject_folder][trial_idx][dev_str] = {}
            IMU_rot_data[subject_folder][trial_idx][dev_str] = {}

            # Extracting data for each sensor (shape: Timestep x axis)
            sensor_acc = np.squeeze(acc_data)[idx]
            sensor_gyr = np.squeeze(gyr_data)[idx]
            sensor_mag = np.squeeze(mag_data)[idx]

            sensor_rot = np.transpose(np.squeeze(rot_data)[idx], (2, 0, 1) )  # Change dim to Timesteps x 3 x 3


            # Handling the mismatch in dimension between the rotation matrices and measurements
            if not sensor_rot.shape[0] == sensor_acc.shape[0]:  # If there is a mismatch
                mismatch_size = sensor_rot.shape[0] - sensor_acc.shape[0]
                if mismatch_size > 0: # If rotation matrices have more timestamps
                    start = mismatch_size // 2
                    sensor_rot = sensor_rot[start : start+sensor_acc.shape[0]]
                    print(f'Rotation Matrices have more timestamps in file {trial_file}, \n'
                          f'trimming off {mismatch_size} datapoints')

                else: # If rotation matrices have less timestamps
                    start = abs(mismatch_size) // 2
                    sensor_acc = sensor_acc[start : start+sensor_rot.shape[0]]
                    sensor_gyr = sensor_gyr[start: start + sensor_rot.shape[0]]
                    sensor_mag = sensor_mag[start: start + sensor_rot.shape[0]]
                    print(f'Rotation Matrices have fewer timestamps in file {trial_file}, \n'
                          f'trimming off {mismatch_size} datapoints')

            # Rotating the accelerometer data and gyroscope data
            rot_acc = np.einsum('bij,bj->bi', sensor_rot, sensor_acc)
            rot_gyr = np.einsum('bij,bj->bi', sensor_rot, sensor_gyr)
            rot_mag = np.einsum('bij,bj->bi', sensor_rot, sensor_mag)
            # These calculations are verified, the rot_ac results in an acceleration of 9.81 in z direction (Gravity) (Positive though)

            # Storing them in the dictionary
            IMU_raw_data[subject_folder][trial_idx][dev_str] = {
                'Acceleration': {
                    'x': sensor_acc[:, 0].tolist(),
                    'y': sensor_acc[:, 1].tolist(),
                    'z': sensor_acc[:, 2].tolist()},

                'Angular Velocity': {
                    'x': sensor_gyr[:, 0].tolist(),
                    'y': sensor_gyr[:, 1].tolist(),
                    'z': sensor_gyr[:, 2].tolist()},

                'Magnetic Field' : {
                    'x': sensor_mag[:, 0].tolist(),
                    'y': sensor_mag[:, 1].tolist(),
                    'z': sensor_mag[:, 2].tolist()}
            }
            IMU_rot_data[subject_folder][trial_idx][dev_str] = {
                'Rotated Acceleration': {
                    'x': rot_acc[:, 0].tolist(),
                    'y': rot_acc[:, 1].tolist(),
                    'z': rot_acc[:, 2].tolist()},

                'Rotated Angular Velocity': {
                    'x': rot_gyr[:, 0].tolist(),
                    'y': rot_gyr[:, 1].tolist(),
                    'z': rot_gyr[:, 2].tolist()},

                'Rotated Magnetic Field': {
                    'x': rot_mag[:, 0].tolist(),
                    'y': rot_mag[:, 1].tolist(),
                    'z': rot_mag[:, 2].tolist()}
            }

# Saving the dictionary to a JSON file
print('Saving the raw data...')
with open(r'C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Data Files\StraightWalking\JSON\IMU_raw.json', 'w') as file:
    json.dump(IMU_raw_data, file, indent=2)
print('Raw data successfully saved')
print('Saving the rotated data...')
with open(r'C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Data Files\StraightWalking\JSON\IMU_rotated.json', 'w') as file:
    json.dump(IMU_rot_data, file, indent=2)
print('Rotated data successfully saved')

