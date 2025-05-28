clc; close all; clear;
%% === Final Variable Initialisation ===
MachineLearningData = struct();

%% === Script Settings ===
PLOT_FRAMES = true;
PLOT_TRANSFER = false;

g = 9.81;
stepThreshold = 0.8; % Fraction of the body force, which will be considered a step
minStepDistance = 0.1;  % Minimum step distance, to be considered a step

%% === Loading Data ===
parentDir = 'C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Data Files\StraightWalking\MatLabCombined';

IMU_ALL = load([parentDir '\IMU.mat']).IMUData;  % Non-rotated IMU signal
VICON_ALL = load([parentDir '\Vicon.mat']).ViconData;
IFS_ALL = load([parentDir '\GRF.mat']).GRFData;
SyncIdx = load([parentDir '\SyncIndices.mat']).syncIndices;
metaData = load([parentDir '\MetaData.mat']).patientMetaData;

patientList = fieldnames(IMU_ALL);
trialList = fieldnames(IMU_ALL.P191008);
%% === Looping over patients ===
for patient = 1:length(patientList)
    patientID = patientList{patient};
    if ~strcmp(patientID, "P191108")  % 1 patient is excluded because 
            % measurement frequencies are not consistent across devices
        bodyWeight = metaData.(patientID).Weight;  % Mass of the patient
        bodyForce = bodyWeight*g;  % Force exerted by gravity
    
        %% === Looping over trials === 
        for trial = 1:length(trialList)
            trialID = trialList{trial};
    
            %% == Storing Measurement Indices == 
            imuStart = SyncIdx.(patientID).(trialID).IMU(1);
            imuStop = SyncIdx.(patientID).(trialID).IMU(2);
            
            grfStart = SyncIdx.(patientID).(trialID).IFS(1);
            grfStop = SyncIdx.(patientID).(trialID).IFS(2);
    
            viconStart = SyncIdx.(patientID).(trialID).Vicon(1);
            viconStop = SyncIdx.(patientID).(trialID).Vicon(2);
    
            %% === Step Detection ===
            leftGRF = IFS_ALL.(patientID).(trialID).Left(grfStart:grfStop, 3) / bodyForce;
            leftStepIndices = findStepIndices(leftGRF, stepThreshold);
            
            rightGRF = IFS_ALL.(patientID).(trialID).Right(grfStart:grfStop, 3) / bodyForce;
            rightStepIndices = findStepIndices(rightGRF, stepThreshold);
            
            %% == Storing Step Positions ==
            leftPos = VICON_ALL.(patientID).(trialID).LF(viconStart:viconStop, 1:2);
            leftPos = leftPos(leftStepIndices, :);
    
            rightPos = VICON_ALL.(patientID).(trialID).RF(viconStart:viconStop, 1:2);
            rightPos = rightPos(rightStepIndices, :);

     
            %% == Computing Step Frames ==
            [leftOrigins, leftRotations] = computeStepFramesV2(leftPos);
            [rightOrigins, rightRotations] = computeStepFramesV2(rightPos);
            
            [sortedIdx, sortMask] = sort([leftStepIndices(1:end-1); rightStepIndices(1:end-1)]);
    
            combinedIndices = [leftStepIndices(1:end-1); rightStepIndices(1:end-1)];
            combinedIndices = combinedIndices(sortMask);
            
            combinedOrigins = [leftOrigins; rightOrigins];
            combinedOrigins = combinedOrigins(sortMask, :);
            
            combinedRotations = cat(3, leftRotations, rightRotations);
            combinedRotations = combinedRotations(:, :, sortMask);

            %% === Plotting Step Frames
            if PLOT_FRAMES == true
                xOrg = combinedOrigins(:, 1);
                yOrg = combinedOrigins(:, 2);
                
                walkDir = squeeze(combinedRotations(1:2, 1, :))';
                latDir = squeeze(combinedRotations(1:2, 2, :))';

                figure
                hold on;
                title(sprintf('StepFrames Patient: %s, Trial: %s', patientID, trialID))
                quiver(xOrg, yOrg, walkDir(:, 1), walkDir(:, 2), 0.3, 'b', 'LineWidth', 1.5, 'MaxHeadSize', 2, 'DisplayName', 'Walking Direction')
                quiver(xOrg, yOrg, latDir(:, 1), latDir(:, 2), 0.3, 'r', 'LineWidth', 1.5, 'MaxHeadSize', 2, 'DisplayName', 'Hip-Hip Direction')
                scatter(xOrg, yOrg, 20, 'k', 'filled', 'DisplayName', 'Origin Step Frame')
                legend show;
                grid on;
                axis equal;
            end

            %% === Cutting trial data ===
            % Vicon Data
            comTrial = VICON_ALL.(patientID).(trialID).COM(viconStart:viconStop, :);
            
            % IMU data
            imuTrial = struct();
            imuPositions = fieldnames(IMU_ALL.(patientID).(trialID));
            for place = 1:length(imuPositions)
                sensorID = imuPositions{place};
                sensorData = IMU_ALL.(patientID).(trialID).(sensorID);
               
                %% -- Rotating the IMU data at the start of the trial --
                startQuat = sensorData.Quaternion(imuStart, :);
                sensorData.Accelerometer = calibrateIMUSignal( ...
                    sensorData.Accelerometer(imuStart:imuStop, :), ...
                    startQuat);
                sensorData.Gyroscope = calibrateIMUSignal( ...
                    sensorData.Gyroscope(imuStart:imuStop, :), ...
                    startQuat);
    
                imuTrial.(sensorID) = rmfield(sensorData, 'Quaternion');
            end
            
            % GRF Data
            grfTrial = IFS_ALL.(patientID).(trialID).Total(grfStart:grfStop, :);

            %% === Looping over steps ===

            %% -- Creating Plot --
            if PLOT_TRANSFER == true
                figure;
                sgtitle(sprintf('Transformed COM Position: %s, trial: %s', patientID, trialID));
                subplot(3, 1, 1); hold on; title('X'); ylabel('X');
                subplot(3, 1, 2); hold on; title('Y'); ylabel('Y');
                subplot(3, 1, 3); hold on; title('Z'); ylabel('Z'); xlabel('Time');
            end

            for k = 2:length(combinedIndices)
                startIdx = combinedIndices(k-1);
                stopIdx = combinedIndices(k);
                
                %% -- Extracting IMU of step --
                imuStep = struct();
                for place = 1:length(imuPositions)
                    sensorID = imuPositions{place};
                    
                    acc = imuTrial.(sensorID).Accelerometer(startIdx:stopIdx, :);
                    gyr = imuTrial.(sensorID).Gyroscope(startIdx:stopIdx, :);
    
                    imuStep.(sensorID) = struct( ...
                        'Accelerometer', acc, ...
                        'Gyroscope', gyr);
                end
    
                %% -- Extracting COM of step --
                comStep = coordinateTransfer( ...
                    comTrial(startIdx:stopIdx, :), ...
                    combinedOrigins(k-1, :), ...
                    combinedRotations(:, :, k-1));
                
                %% -- Extracting GRF of step --
                % Should rotate this, TO DO
                grfStep = grfTrial(startIdx:stopIdx, :);
                
    
                %% === Storing Data ===
                stepID = sprintf('S%d', k-1);
                MachineLearningData.(patientID).(trialID).(stepID) = struct( ...
                    'IMU', imuStep, ...
                    'COM', comStep, ...
                    'GRF', grfStep);
                
                %% === Plotting Options ===
                if PLOT_TRANSFER == true % Plot only first step
                   x = comStep(:, 1);
                   y = comStep(:, 2);
                   z = comStep(:, 3);
            
                   subplot(3, 1, 1)
                   plot(x)
        
                   subplot(3, 1, 2)
                   plot(y)
        
                   subplot(3, 1, 3)
                   plot(z)              
                end    
            end
        end
    end
end

jsonText = jsonencode(MachineLearningData, 'PrettyPrint', true);
fid = fopen('C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Data Files\StraightWalking\Python\MachineLearningData.json', 'w');
fwrite(fid, jsonText, 'char');
fclose(fid);


function unitVecs = normalizeVectors(vecs)
    magnitudes = sqrt(sum(vecs.^2, 2));
    unitVecs = vecs ./ magnitudes;
end


function stepIndices = findStepIndices(grfSignal, stepThreshold)
    grfSignal(grfSignal > stepThreshold) = stepThreshold;
    stepSignal = abs(grfSignal - stepThreshold) < 1e-6;
    stepSignal = diff([0; stepSignal; 0]);
    stepsStart = find(stepSignal == 1);
    stepsEnd = find(stepSignal == -1);
    stepIndices = round((stepsStart + stepsEnd) / 2);
end


function [Origins, Rotations] = computeStepFramesV2(stepPositions)
    nSteps = size(stepPositions, 1) - 1;
    zCol = zeros(nSteps, 1);

    stepDiff = diff(stepPositions);
    Origins = [stepPositions(1:end-1, :), zCol];

    walkingDirection = [normalizeVectors(stepDiff), zCol];
    lateralDirection = [-walkingDirection(:, 2), walkingDirection(:, 1), zCol];
    verticalDirection = [zCol, zCol, ones(nSteps, 1)];

    Rotations = zeros(3, 3, nSteps);
    for i = 1:nSteps
        u = walkingDirection(i, :)';
        v = lateralDirection(i, :)';
        w = verticalDirection(i, :)';
        R = [u, v, w];
        Rotations(:, :, i) = R;
    end
end


function transferredPosition = coordinateTransfer(orgPos, Origin, Rotation)
    transferredPosition = zeros(size(orgPos));
    nTimesteps = length(orgPos);
    Origin = repmat(Origin, nTimesteps, 1);
    Rotation = repmat(Rotation, 1, 1, nTimesteps);
    
    for t = 1:nTimesteps
        transferredPosition(t, :) = (orgPos(t, :) - Origin(t, :)) * Rotation(:, :, t)';  
    end
end


function calibratedSignal = calibrateIMUSignal(imuSignal, quaternion)
    signalLength = length(imuSignal);
    quats = repmat(quaternion, signalLength, 1);
    
    quatsConj = quats;
    quatsConj(:, 2:4) = -quats(:, 2:4);
    
    signalQuat = [zeros(signalLength, 1), imuSignal];

    calibratedSignal = quatmultiply(quatmultiply(quats, signalQuat), quatsConj);
    calibratedSignal = calibratedSignal(:, 2:4);
end