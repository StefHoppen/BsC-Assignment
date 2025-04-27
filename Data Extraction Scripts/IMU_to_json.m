clc
close all 
clear
%% Settings
ROTATE = false;  % If accelerations and gyroscope measurements should be rotated to the global coordinate frame
MAGNETOMETER = false;  % If magnetometer measurments should be included in the json file

%% CODE
output_struct = struct();

parentDir = 'C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Data Files\StraightWalking\MatLab\IMU';
SensorIDs = ["LeftFoot", "RightFoot", "Sternum", "Pelvis"];


% Get list of subfolders (excluding '.' and '..')
folderList = dir(parentDir);
folderList = folderList([folderList.isdir] & ~ismember({folderList.name}, {'.', '..'}));

for i = 1:length(folderList)
    folderName = folderList(i).name;
    patientID = matlab.lang.makeValidName(['P_' folderName]);
    % Creating a path to the folder
    fullFolderPath = fullfile(parentDir, folderName);

    % Get all .mat files in this folder
    matFiles = dir(fullfile(fullFolderPath, '*.mat'));

    for j = 1:length(matFiles)
        trialID = matlab.lang.makeValidName(['T' num2str(j)]);
        matFileName = matFiles(j).name;
        fullMatPath = fullfile(fullFolderPath, matFileName);

        % Loading the .mat file
        raw_data = load(fullMatPath).rawdata;
        test = load(fullMatPath);

        % Looping over each sensor
        for k = 1: length(SensorIDs)
            acc = raw_data.acc{k};
            gyr = raw_data.gyr{k};
            mag = raw_data.mag{k};
            
            if ROTATE == true
                % Getting quaternions
                quats = raw_data.q_ls_xda{k};
                % Creating quaternion conjugates
                quats_conj = quats;
                quats_conj(:, 2:4) = -quats(:, 2:4);

                % Creating quaternions from vectors
                acc_quat = [zeros(size(acc, 1), 1), acc];
                gyr_quat = [zeros(size(acc, 1), 1), gyr];
                mag_quat = [zeros(size(acc, 1), 1), mag];

                % Rotating the vectors using the quaternions
                acc = quatmultiply(quatmultiply(quats, acc_quat), quats_conj);
                gyr = quatmultiply(quatmultiply(quats, gyr_quat), quats_conj);
                mag = quatmultiply(quatmultiply(quats, mag_quat), quats_conj);

                % Cutting of the scalar part
                acc = acc(:, 2:4);
                gyr = gyr(:, 2:4);
                mag = mag(:, 2:4);
            end
            sensorID = char(SensorIDs(k));
            
            accData = struct( ...
                'x', acc(:, 1), ...
                'y', acc(:, 2), ...
                'z', acc(:, 3));
            gyrData = struct( ...
                'x', gyr(:, 1), ...
                'y', gyr(:, 2), ...
                'z', gyr(:, 3));
            
            magData = struct( ...
                'x', mag(:, 1), ...
                'y', mag(:, 2), ...
                'z', mag(:, 3));
            
            if MAGNETOMETER == true
                sensorData = struct( ...
                    'Accelerometer', accData, ...
                    'Gyroscope', gyrData, ...
                    'Magnetometer', magData);
            else
                sensorData = struct( ...
                    'Accelerometer', accData, ...
                    'Gyroscope', gyrData);
            end



            output_struct.(patientID).(trialID).(sensorID) = sensorData;
        end
    end
end

% Encode and save
jsonStr = jsonencode(output_struct, 'PrettyPrint', true);  % Nice formatting
fid = fopen('IMU_Raw.json', 'w');
if fid == -1
    error('Cannot create JSON file');
end
fwrite(fid, jsonStr, 'char');
fclose(fid);
