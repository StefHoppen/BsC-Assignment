clc
close all 
clear
%% Settings
ROTATE = true;  % If accelerations and gyroscope measurements should be rotated to the global coordinate frame
MAGNETOMETER = false;  % If magnetometer measurments should be included in the json file

%% CODE
IMUData = struct();
RightFootData = struct();

parentDir = 'C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Data Files\StraightWalking\MatLab\IMU';
SensorIDs = ["LeftFoot", "RightFoot", "Sternum", "Pelvis"];


% Get list of subfolders (excluding '.' and '..')
folderList = dir(parentDir);
folderList = folderList([folderList.isdir] & ~ismember({folderList.name}, {'.', '..'}));

for i = 1:length(folderList)
    folderName = folderList(i).name;
    patientID = matlab.lang.makeValidName(['P' folderName]);
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
            
            % If sensorID is right foot, also save it to a different
            % struct (needed for syncing)
            if strcmp(sensorID, 'RightFoot')
                RightFootData.(patientID).(trialID) = accData;
            
            end

            end



            IMUData.(patientID).(trialID).(sensorID) = sensorData;
        end
    end
end
%% Saving to .mat file
save("C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Data Files\StraightWalking\MatLabCombined\IMU_Rotated.mat", ...
    'IMUData')
save("C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Data Files\StraightWalking\MatLabCombined\IMU_RF.mat", ...
    'RightFootData')

