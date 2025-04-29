%% MEASUREMENT SYNCING SCRIPT
clc
close all 
clear

PLOT_ORG_SIGNAL = false;
VISUAL_CHECK = false;
StartStopIDX = struct();
%% Loading identifiers
% Identifier is 'raise right foot', thus we must extract the y-acceleration
% of:
% 1. Right ForceShoe
% 2. Right Heel Marker VICON
% 3. Right foot IMU
% Then these are matched and the corresponding indexes are stored
folderpath = 'C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Data Files\StraightWalking\MatLabCombined\SyncingFeatures(RightFoot)';

% Loading ForceShoe Right Foot Accelerations
IFS_Acc = load([folderpath '\IFS_RF.mat']).RightFootData;

% Loading VICON Right Foot Position
VICON_Pos = load([folderpath '\VICON_RF.mat']).RightFootData;

% Loading IMU Right Foot Accelerations
IMU_Acc = load([folderpath '\IMU_RF']).RightFootData;

%% Filtering constants (Just for syncing)
sampling_f = 100; % Sampling frequency
cutoff_f = 2; % Cutoff frequency

g = 9.81;

%% Looping over patients
patientList = fieldnames(VICON_Pos);
trialList = fieldnames(VICON_Pos.P191008);
for patient = 1:length(patientList)
    patientID = patientList{patient};
    
    %% Loop over trials
    for trial = 1:length(trialList)
        trialID = trialList{trial};
        %% Get the vertical values of all measurements
        IFS = IFS_Acc.(patientID).(trialID).z + g;
        VICON = VICON_Pos.(patientID).(trialID).z;
        IMU = -IMU_Acc.(patientID).(trialID).z + g;

        %% Filtering Acceleration Data
        IFS_filtered = lowpass(IFS, cutoff_f, sampling_f);
        IMU_filtered = lowpass(IMU, cutoff_f, sampling_f);
        if PLOT_ORG_SIGNAL == true
            VICON_VEL = diff(VICON) / 0.01;
            VICON_VEL = lowpass(VICON_VEL, 0.5, sampling_f);
            VICON_ACC = diff(VICON_VEL) / 0.01;
            VICON_ACC = lowpass(diff(VICON_ACC), 0.5, sampling_f);
            
            figure;
            subplot(3, 1, 1);  % (rows, columns, position)
            hold on;
            plot(VICON, 'b', 'LineWidth', 1.5, 'DisplayName', 'VICON POS');
            %plot(VICON_VEL, 'g', 'LineWidth', 1.5, 'DisplayName', 'VICON VEL');
            %plot(VICON_ACC, 'r', 'LineWidth', 1.5, 'DisplayName', 'VICON ACC');
            legend show;
            xlim([1500 3500])
            
            subplot(3, 1, 2);  % (rows, columns, position)
            plot(IMU_filtered, 'b', 'LineWidth', 1.5, 'DisplayName', 'IMU');
            xlim([1500 3500])

            subplot(3, 1, 3);  % (rows, columns, position)
            plot(IFS_filtered, 'r', 'LineWidth', 1.5, 'DisplayName', 'IFS');
            xlim([1500 3500])

            legend show;
            title('Unsynced signals'); 
        end

      
        %% Syncing
        % For the VICON signal, the start and end signal are always the two
        % largest peaks
        [vicon_peaks, vicon_locs] = findpeaks(VICON, NPeaks=2, MinPeakHeight=300, MinPeakDistance=1000);
        viconStart = vicon_locs(1);
        viconStop = vicon_locs(end);
        viconLength = viconStop - viconStart;

        
        % For the IMU and IFS signal, the peak of the VICON signal
        % corresponds to the positive peaks (Coordinate frame is opposite)
        [imuPeaks, imuLocs] = findpeaks(IMU_filtered, minPeakHeight=3);
        imuStart = imuLocs(1);
        imuStop = imuLocs(end);
        imuLength = imuStop - imuStart;

        [ifsPeaks, ifsLocs] = findpeaks(IFS_filtered, minPeakHeight=3);
        ifsStart = ifsLocs(1);
        ifsStop = ifsLocs(end);
        ifsLength = ifsStop - ifsStart;
        
        % We assume the VICON data is the ground truth, thus we will sync
        % everything to that.
        %% IMU Syncing
        diffIMU = viconLength - imuLength;
        if diffIMU > 0  % ViconLength is larger
            imuStart = imuStart - diffIMU;
        elseif diffIMU < 0  % ViconLength is smaller
            imuStop = imuStop - diffIMU;
        else % Perfect match
            disp('IMU and Vicon: Perfect Match')
        end

        %% IFS Syncing
        diffIFS = viconLength - ifsLength;
        if diffIFS > 0  % ViconLength is larger
            ifsStart = ifsStart - ceil(abs(diffIFS)/2);
            ifsStop = ifsStop + floor(abs(diffIFS)/2);
        elseif diffIFS < 0  % ViconLength is smaller
            ifsStart = ifsStart + ceil(abs(diffIFS)/2);
            ifsStop = ifsStop - floor(abs(diffIFS)/2);
        else % Perfect match
            disp('IFS and Vicon: Perfect Match')
        end

        %% Visual Check
        if VISUAL_CHECK == true
            imuSynced = IMU(imuStart:imuStop);
            ifsSynced = IFS(ifsStart:ifsStop);
            viconSynced = VICON(viconStart:viconStop);
    
            figure;
            hold on;
            plot(viconSynced, 'g', 'LineWidth', 1.5, 'DisplayName', 'VICON');
            plot(imuSynced, 'b', 'LineWidth', 1.5, 'DisplayName', 'IMU');
            plot(ifsSynced, 'r', 'LineWidth', 1.5, 'DisplayName', 'IFS');
            legend show;
            title('Press "y" to accept and continue, "n" to reject and stop ');
            
            while true
                response = input('Accept signals? (y/n): ', 's');
        
                if strcmpi(response, 'y')
                    close(gcf);
                    disp('Accepted. Continuing...');
                    break;  % exit the loop and continue the script
                elseif strcmpi(response, 'n')
                    disp('Rejected. Stopping script.');
                    return; % exit the script
                else
                    disp('Invalid input. Please enter ''y'' or ''n''.');
                end
            end
        end
        
        %% STORING MEASUREMENT BEGIN AND END INDEX
        StartStopIDX.(patientID).(trialID) = struct( ...
            'Vicon', [viconStart, viconStop], ...
            'IMU', [imuStart, imuStop], ...
            'IFS', [ifsStart, ifsStop]);
    end
end
save("C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Data Files\StraightWalking\MatLabCombined\SyncIDX.mat", ...
    'StartStopIDX')
