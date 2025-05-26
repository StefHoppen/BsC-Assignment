clc; close all; clear;

%% === Script Settings ===
PLOT_ORG_SIGNAL = false;  % Plot the original signals, prior to the synchronisation
PLOT_TOTAL_SYNC = true;  % Plot the signal after the synchronization

%% === Variable Initialisation ===
syncIndices = struct();  % Struct which stores the beginning and end indices of each measurment type

%% === Loading Data ===
folderpath = 'C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Data Files\StraightWalking\MatLabCombined';

IMU_ALL = load([folderpath '\IMU.mat']).IMUData;  % Non-rotated IMU signal
VICON_ALL = load([folderpath '\Vicon.mat']).ViconData;
IFS_ALL = load([folderpath '\GRF.mat']).GRFData;
IFS_ACC_RF = load([folderpath '\IFS_ACC_RF.mat']).RightFootData;

%% === Looping over patients ===
patientList = fieldnames(VICON_ALL);
trialList = fieldnames(VICON_ALL.P191008);
for patient = 1:length(patientList)
    patientID = patientList{patient};

    %% === Looping over trials
    for trial = 1:length(trialList)
        trialID = trialList{trial};
        
        %% -- Storing measurements --
        grfTrial = IFS_ALL.(patientID).(trialID);
        viconTrial = VICON_ALL.(patientID).(trialID);
        imuTrial = IMU_ALL.(patientID).(trialID);
        
        % Storing right foot vertical measurement in different variables 
        % for sake of simplicity 
        ifsRF = IFS_ACC_RF.(patientID).(trialID)(3, :)';  % Vertical acceleration
        viconRF = viconTrial.RF(:, 3);  % Get vertical displacement of right foot

        imuRF = rotateGlobal(imuTrial.RF.Accelerometer, imuTrial.RF.Quaternion);
        imuRF = imuRF(:, 3); % Get only vertical signal


        %% -- Plotting Original Signal Shape --
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
        
        %% === Synchronisation ===
        % The previous plot showed that for Patient 191108 the measurement
        % frequency is not constant across measurements. Thus I will
        % exclude it for now
        if ~strcmp(patientID, "P191108")            
            % Making 0 centred, better for correlation
            viconRF = viconRF - mean(viconRF);
            imuRF = imuRF - mean(imuRF);
            ifsRF = ifsRF - mean(ifsRF);

            %% == Aligning IFS and IMU ==
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
            
            %% == Aligning Vicon and IMU ==
            % Alligning the VICON to the IMU (Same process as IFS/IMU)
            [corrVals, lagVals] = xcorr(viconRF, imuRF);
            [~, maxIndex] = max(abs(corrVals));
            viconLag = lagVals(maxIndex);

            %% -- Finding start/stop indices --
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
            
            %% === Storing start/stop indices in struct
            trialIndices = struct( ...
                'IMU', [imuStart, imuStop], ...
                'IFS', [ifsStart, ifsStop], ...
                'Vicon', [viconStart, viconStop]);
            syncIndices.(patientID).(trialID) = trialIndices;
        end
    end
end

save("C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Data Files\StraightWalking\MatLabCombined\SyncIndices.mat", ...
    'syncIndices')




%% ==== Function: rotateGlobal ====
function globalSignal = rotateGlobal(signal, quats)
    % Getting quaternion conjugates
    quatsConj = quats;
    quatsConj(:, 2:4) = -quats(:, 2:4);
    
    % Converting signal to quaternion 
    signalQuat = [zeros(size(signal, 1), 1), signal];
    
    % Rotating signal with quaternions
    globalSignal = quatmultiply(quatmultiply(quats, signalQuat), quatsConj);
    globalSignal = globalSignal(:, 2:4); % Cutting of quaternion scalar
end