%% Measurement Syncing Script
clc
close all
clear

%% Script Settings
PLOT_ORG_SIGNAL = false;  % Plot the original signals, prior to the synchronisation
PLOT_TOTAL_SYNC = true;  % Plot the signal after the synchronization and cut

%% Variable Initialisation
SyncedMeasurements = struct();

%% Loading Data
folderpath = 'C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Data Files\StraightWalking\MatLabCombined';

% Machine Learning Inputs
IMU_ALL = load([folderpath '\IMU_Rotated.mat']).IMUData;
VICON_ALL = load([folderpath '\Vicon.mat']).ViconData;
IFS_ALL = load([folderpath '\GRF.mat']).GRFData;

% Syncing Inputs
IFS_ACC_RF = load([folderpath '\IFS_ACC_RF.mat']).RightFootData;

%% Looping over patients
patientList = fieldnames(VICON_ALL);
trialList = fieldnames(VICON_ALL.P191008);
for patient = 1:length(patientList)
    patientID = patientList{patient};


    %% Loop over trials
    for trial = 1:length(trialList)
        trialID = trialList{trial};
        
        % Getting measurements of the trial
        grfTrial = IFS_ALL.(patientID).(trialID);
        viconTrial = VICON_ALL.(patientID).(trialID);
        imuTrial = IMU_ALL.(patientID).(trialID);

        % Storing right foot vertical measurement in different variables 
        % for sake of simplicity 
        ifsRF = IFS_ACC_RF.(patientID).(trialID)(3, :)';
        viconRF = viconTrial.RF(:, 3);
        imuRF = imuTrial.RF.Accelerometer(:, 3);
        
        %% Plotting Original Signal Shape
        if PLOT_ORG_SIGNAL == true
            longestMeasurement = max([length(ifsRF), length(imuRF), length(viconRF)]);
            
            figure;
            sgtitle(sprintf('Signal for Patient: %s, trial: %s ', patientID, trialID));
            subplot(3, 1, 1);  % (rows, columns, position)
            plot(viconRF, 'b', 'LineWidth', 1.5, 'DisplayName', 'VICON POS');
            legend show;
            xlim([0 longestMeasurement])
            
            subplot(3, 1, 2);  % (rows, columns, position)
            plot(imuRF, 'b', 'LineWidth', 1.5, 'DisplayName', 'IMU');
            legend show;
            xlim([0 longestMeasurement])

            subplot(3, 1, 3);  % (rows, columns, position)
            plot(ifsRF, 'r', 'LineWidth', 1.5, 'DisplayName', 'IFS');
            xlim([0 longestMeasurement])
            legend show;
        end


        %% Synchronisation
        % The previous plot showed that for Patient 191108 the measurement
        % frequency is not constant across measurements. Thus I will
        % exclude it for now

        if ~strcmp(patientID, "P191108")            
            % Making 0 centred, better for correlation
            viconRF = viconRF - mean(viconRF);
            imuRF = imuRF - mean(imuRF);
            ifsRF = ifsRF - mean(ifsRF);
            
            %% ALIGNING IFS and IMU
            % First we allign the IFS to the IMU

            % Calculating signal correlation
            [corrVals, lagVals] = xcorr(ifsRF, imuRF);
            % Finding out the lag with the maximum correlation
            [~, maxIndex] = max(abs(corrVals));

            % Storing this best lag value
            ifsLag = lagVals(maxIndex);
            % If ifsLag is POSITIVE, the IFS signal is IN FRONT of the IMU
            % signal. Thus we should move it back.

            % If ifsLag is NEGATIVE, the IFS signal is BEHIND the IMU
            % signal. Thus we move it forward.
            
            %% ALIGNING VICON and IMU
            % Alligning the VICON to the IMU (Same process as IFS/IMU)
            [corrVals, lagVals] = xcorr(viconRF, imuRF);
            [~, maxIndex] = max(abs(corrVals));
            viconLag = lagVals(maxIndex);

            %% FINDING MEASUREMENT INDEXES
            % The start and stop of each measurement is manually selected
            % using a plotting method.
            figure;
            plot(viconRF);
            title('Click on the peaks you want. Press Enter when done.');
            [Loc, ~] = ginput(2);
            close(gcf);

            viconStart = fix(Loc(1));
            viconStop = fix(Loc(end));  % Converting also to integers

            % Since all signals are synced to the IMU (worked best) but
            % the signal start is given by the VICON, this piece of code is
            % a bit chaotic. But it works.
            
            % If viconLag is POSITIVE:
            %   Vicon is in FRONT of the IMU -> imuIDX < viconIDX
            % If viconLag is NEGATIVE:
            %   Vicon is BEHIND IMU -> imuIDX > viconIDX 
            imuStart = viconStart - viconLag;
            imuStop = viconStop - viconLag;

            % If ifsLag is POSITIVE:
            %   IFS is IN FRONT of the IMU -> imuIDX < ifsIDX
            % If ifsLag is NEGATIVE:
            %   IFS is BEHIND the IMU -> ifsIDX > imuIDX
            ifsStart = imuStart + ifsLag;
            ifsStop = imuStop + ifsLag;
            

            %% CUTTING ALL SIGNALS
            cutMeasurement = struct();
            % Cutting VICON
            cutMeasurement.VICON.LF = viconTrial.LF(viconStart:viconStop, :);
            cutMeasurement.VICON.RF = viconTrial.RF(viconStart:viconStop, :);
            cutMeasurement.VICON.COM = viconTrial.COM(viconStart:viconStop, :);

            % Cutting GRFs
            cutMeasurement.GRF.Total = grfTrial.Total(ifsStart:ifsStop, :);
            cutMeasurement.GRF.LF = grfTrial.Left(ifsStart:ifsStop, :);
            cutMeasurement.GRF.RF = grfTrial.Right(ifsStart:ifsStop, :);

            % Cutting IMU
            sensorPosList = fieldnames(imuTrial);
            sensorTypeList = fieldnames(imuTrial.LF);
            for i = 1:length(sensorPosList)
                for j = 1:length(sensorTypeList)
                sensorPos = sensorPosList{i};
                sensorType = sensorTypeList{j};
                
                cutMeasurement.IMU.(sensorPos).(sensorType) = ...
                    imuTrial.(sensorPos).(sensorType)(imuStart:imuStop, :);
                end
            end

            SyncedMeasurements.(patientID).(trialID) = cutMeasurement;
            

            %% Plotting if syncing is correct
            if PLOT_TOTAL_SYNC == true
                figure;
                measurementLength = imuStop - imuStart;
                sgtitle(sprintf('Synchronized Signal for Patient: %s, trial: %s ', patientID, trialID));
                subplot(3, 1, 1);  % (rows, columns, position)
                plot(viconRF(viconStart:viconStop, :), 'b', 'LineWidth', 1.5, 'DisplayName', 'VICON POS');
                legend show;
                xlim([0 measurementLength])
                
                subplot(3, 1, 2);  % (rows, columns, position)
                plot(imuRF(imuStart:imuStop, :), 'b', 'LineWidth', 1.5, 'DisplayName', 'IMU');
                legend show;
                xlim([0 measurementLength])
    
                subplot(3, 1, 3);  % (rows, columns, position)
                plot(ifsRF(ifsStart:ifsStop, :), 'r', 'LineWidth', 1.5, 'DisplayName', 'IFS');
                xlim([0 measurementLength])
                legend show;
            end
        end
    end
end
save("C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Data Files\StraightWalking\MatLabCombined\SynchronizedMeasurements.mat", ...
    'SyncedMeasurements')