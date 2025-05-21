clc
close all 
clear
%% Settings
ROTATE = true;  % If accelerations and gyroscope measurements should be rotated to the global coordinate frame
MAGNETOMETER = false;  % If magnetometer measurments should be included in the json file
FILTERING = true; % If signals should be filtered

%% Filtering settings
fPassAcc = 1; % Pass frequency for the low-pass filter (acceleration) (hz)
fPassGyro = 3; % Pass frequency for the low-pass filter (gyroscope) (hz)
fSampling = 100; %Hz

%% CODE
IMUData = struct();  % Initialising struct for IMU data
parentDir = 'C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Data Files\StraightWalking\MatLab\IMU'; % Directory raw files are stored
SensorIDs = ["LF", "RF", "Ste", "Pel"];  % Sensor positions

% Get list of subfolders (excluding '.' and '..')
folderList = dir(parentDir);
folderList = folderList([folderList.isdir] & ~ismember({folderList.name}, {'.', '..'}));

% Loop over folders (Patients)
for i = 1:length(folderList)
    folderName = folderList(i).name;
    patientID = matlab.lang.makeValidName(['P' folderName]);
    % Creating a path to the folder
    fullFolderPath = fullfile(parentDir, folderName);

    % Get all .mat files in this folder
    matFiles = dir(fullfile(fullFolderPath, '*.mat'));
    
    % Loop over files (Trials)
    for j = 1:length(matFiles)
        trialID = matlab.lang.makeValidName(['T' num2str(j)]);
        matFileName = matFiles(j).name;
        fullMatPath = fullfile(fullFolderPath, matFileName);

        % Loading the .mat file
        raw_data = load(fullMatPath).rawdata;

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

            if FILTERING == true
                % Looping over axis
                for m = 1:3
                    % Applying low pass filter on each measurement type
                    acc(:, m) = lowpass(acc(:, m), fPassAcc, fSampling);
                    gyr(:, m) = lowpass(gyr(:, m), fPassGyro, fSampling); 
                end  
            end
            sensorID = char(SensorIDs(k)); % Converting sensorID to 'char' type
            if MAGNETOMETER == true
                sensorData = struct( ...
                    'Accelerometer', acc, ...
                    'Gyroscope', gyr, ...
                    'Magnetometer', mag);
            else
                sensorData = struct( ...
                    'Accelerometer', acc, ...
                    'Gyroscope', gyr);
            end

            IMUData.(patientID).(trialID).(sensorID) = sensorData;  % Storing data in 1 struct
        end
    end
end
%% Saving to .mat file
save("C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Data Files\StraightWalking\MatLabCombined\IMU_Rotated.mat", ...
    'IMUData')

