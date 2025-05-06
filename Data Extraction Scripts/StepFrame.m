clc
close all 
clear
%% Script settings
PLOT_STEPS = false;
PLOT_MEASUMRENTS = false;

g = 9.81;
stepThreshold = 0.8;  % Threshold when the GRF is considered a step

%% Loading Data
parentDir = 'C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Data Files\StraightWalking\MatLabCombined';
allData = load([parentDir '\SynchronizedMeasurements.mat']).SyncedMeasurements;
metaData = load([parentDir '\MetaData.mat']).patientMetaData;

%% Initialising Final Variable
MachineLearningData = struct();

%% Looping over Patients
% Getting patients and trials information
patientList = fieldnames(allData);
trialList = fieldnames(allData.P191008);

for patient = 1:length(patientList)
    patientID = patientList{patient};
    bodyWeight = metaData.(patientID).Weight;
    bodyForce = bodyWeight*g;

    %% Looping over trials
    for trial = 1:length(trialList)
        trialID = trialList{trial};
        %% Detecting steps
        % Getting GRF data as a fraction of the body mass
        leftGRF = allData.(patientID).(trialID).GRF.LF(:, 3) / bodyForce;
        leftGRF(leftGRF > stepThreshold) = stepThreshold;
        rightGRF = allData.(patientID).(trialID).GRF.RF(:, 3) / bodyForce;
        rightGRF(rightGRF > stepThreshold) = stepThreshold;

        isStepLeft = abs(leftGRF - 0.8) < 1e-6; % Tolerance for precision point errors
        isStepRight = abs(rightGRF - 0.8) < 1e-6;

        dLeft = diff([0; isStepLeft; 0]);
        dRight = diff([0; isStepRight; 0]);

        startsLeft = find(dLeft == 1);
        endsLeft = find(dLeft == -1) - 1;
        stepsLeft = round((startsLeft + endsLeft) /2);

        startsRight = find(dRight == 1);
        endsRight = find(dRight == -1) - 1;
        stepsRight = round((startsRight + endsRight) / 2);
        
        %% Assigining Step Frame to each timestep
        % Extracting Step Position from VICON
        leftPos = allData.(patientID).(trialID).VICON.LF;  % X,Y position
        rightPos = allData.(patientID).(trialID).VICON.RF;        
        measurementLength = length(rightPos);

        % Calculating walking direction
        leftStepPos = leftPos(stepsLeft, 1:2);
        rightStepPos = rightPos(stepsRight, 1:2);


        leftDelta = diff(leftStepPos, 1, 1);
        leftNorms = sqrt(sum(leftDelta.^2, 2));
        leftVec = leftDelta ./ leftNorms;

        rightDelta = diff(rightStepPos, 1, 1);
        rightNorms = sqrt(sum(rightDelta.^2, 2));
        rightVec = rightDelta ./ rightNorms;

        % Creating Variables
        Origins = nan(measurementLength, 3);
        Origins(:, 3) = metaData.(patientID).LegLength* 1000; % Convert to mm

        Rotations = nan(3, 3, measurementLength);

        % Looping over step frames
        combinedSteps = sort([stepsLeft; stepsRight]);
        
        for j = 1:length(combinedSteps)
            stepIdx = combinedSteps(j);
            
            % Calculting Origin and Rotation matrix
            if ismember(stepIdx, stepsLeft)
                % If the step originates from the left foot
                leftIdx = find(stepsLeft == stepIdx);
                Org = leftStepPos(leftIdx, :);
                if leftIdx == length(stepsLeft)
                    % Final step in sequence
                    R = nan(3, 3);
                else
                    walkingDir = [leftVec(leftIdx, :) 0];
                    hipDir = [walkingDir(2) -walkingDir(1) 0];
                    upDir = [0 0 1];
                    R = [walkingDir(:), hipDir(:), upDir(:)];
                end

            else
                % If the step originates from the right foot
                rightIdx = find(stepsRight == stepIdx);
                Org = rightStepPos(rightIdx, :);

                if rightIdx == length(stepsRight)
                    R = nan(3, 3);
                else
                    walkingDir = [rightVec(rightIdx, :) 0];
                    hipDir = [walkingDir(2) -walkingDir(1) 0];
                    upDir = [0 0 1];
                    R = [walkingDir(:), hipDir(:), upDir(:)];    
                end
            end
           
            % Assigning each Origin and Rotation matrix to timesteps
            if j == length(combinedSteps)
                % Final step
                N = measurementLength-stepIdx;
            else
                N = combinedSteps(j+1) - stepIdx;
                Org = repmat(Org, N+1, 1);
                R = repmat(R, 1, 1, N+1);

                Origins(stepIdx:stepIdx+N, 1:2) = Org;
                Rotations(:, :, stepIdx:stepIdx+N) = R;

            end
        end
        
        %% Cutting Measurements
        % We again cut the measurement to only include the measurements for
        % which a step frame can be made.
        minIdx = min(combinedSteps);
        maxIdx = min(max(stepsLeft), max(stepsRight))-1;
        
        Rotations = Rotations(:, :, minIdx:maxIdx);
        Origins = Origins(minIdx:maxIdx, :);
        
        % The measurements are not cut perfectly.
        % Ideally we only use measurements of the walking, not from
        % standing still. Therefore, all steps where the walking direction 
        % has a y component > 0.5 (30 degrees) are excluded from the measurement.
        
        % DO SOMETHING HERE
        
        if PLOT_STEPS == true
            x = Origins(:, 1);
            y = Origins(:, 2);

            xBar = squeeze(Rotations(1:2, 1, :))';
            yBar = squeeze(Rotations(1:2, 2, :))';
            
            figure;
            hold on;
            title('Step Frames')
            quiver(x, y, xBar(:, 1), xBar(:, 2), 2, 'b', 'LineWidth', 1.5, 'MaxHeadSize', 2, 'DisplayName', 'Walking Direction')
            quiver(x, y, yBar(:, 1), yBar(:, 2), 2, 'r', 'LineWidth', 1.5, 'MaxHeadSize', 2, 'DisplayName', 'Hip-Hip Direction')
            scatter(x, y, 20, 'k', 'filled', 'DisplayName', 'Origin Step Frame')
            legend show;
            grid on;
            axis equal;
        end
        
        
        %% TRANSFERRING COM POSITION TO STEP FRAME
        comPos = allData.(patientID).(trialID).VICON.COM(minIdx:maxIdx, :);
        
        transferredCOM= zeros(size(comPos));
        for k = 1:length(comPos)
            transferredCOM(k, :) = ((comPos(k, :) - Origins(k,:)) * Rotations(:, :, k))*0.1 ; % Position in cm
        end
        
        % Plotting the results over time
        if PLOT_MEASUMRENTS == true
            time = (0:length(transferredCOM)-1) * 0.01;
            figure;
            sgtitle(sprintf('Transformed COM Position: %s, trial: %s ', patientID, trialID));

            subplot(3, 1, 1)
            plot(time, transferredCOM(:, 1))
            xlabel('Time (s)')
            ylabel('Anteroposterior Position (cm)')

            subplot(3, 1, 2)
            plot(time, transferredCOM(:, 2))
            xlabel('Time (s)')
            ylabel('Mediolateral Position (cm)')

            subplot(3, 1, 3)
            plot(time, transferredCOM(:, 3))
            xlabel('Time (s)')
            ylabel('Vertical Position (cm)')
        end
        
        %% Preparing the Measurements for Python
        % Ideally we want JSON format, due to the tree like nature of
        % experiments.
        GRF = allData.(patientID).(trialID).GRF.Total(minIdx:maxIdx, :);
        FreeGRF = GRF(:, 3) - bodyForce;
        GRF(:, 3) = FreeGRF;
        
        COM = transferredCOM;
        
        IMU = allData.(patientID).(trialID).IMU;
        sensorPositions = fieldnames(IMU);
        sensorTypes = fieldnames(IMU.(sensorPositions{1}));
        for posIdx = 1:length(sensorPositions)
            sensorPos = sensorPositions{posIdx};
            for typeIdx = 1:length(sensorTypes)
                sensorType = sensorTypes{typeIdx};
                IMU.(sensorPos).(sensorType) = IMU.(sensorPos).(sensorType)(minIdx:maxIdx, :);
                
                % Removing g from accelerations
                if strcmp(sensorType, 'Accelerometer')
                    IMU.(sensorPos).(sensorType)(:, 3) = IMU.(sensorPos).(sensorType)(:, 3) - g;
                end
            end
        end
        MachineLearningData.(patientID).(trialID) = struct( ...
            'GRF', GRF, ...
            'COM', COM, ...
            'IMU', IMU);
    end
end
jsonText = jsonencode(MachineLearningData, 'PrettyPrint', true);
fid = fopen('C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Data Files\StraightWalking\Python\MachineLearningData.json', 'w');
fwrite(fid, jsonText, 'char');
fclose(fid);
