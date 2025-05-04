clear; close all; clc
addpath("Functions")

% Specify main path
parentDir = 'C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Data Files\StraightWalking\MatLab\ForceShoe';

% Get list of patient subfolders (excluding '.' and '..')
folderList = dir(parentDir);
folderList = folderList([folderList.isdir] & ~ismember({folderList.name}, {'.', '..'}));

GRFData = struct();
RightFootData = struct();

%% Looping over patients
for i = 1:length(folderList)
    folderName = folderList(i).name;
    patientID = matlab.lang.makeValidName(['P' folderName]);

    % Creating path to folder
    folderPath = fullfile(parentDir, folderName);

    % Getting all trial folders in this patient subfolder
    trailList = dir(folderPath);
    
    % Getting offset values
    offsetName = trailList(~[trailList.isdir]).name;
    offsetPath = fullfile(folderPath, offsetName);
    IFSoffset = load(offsetPath).IFSoffsets;
    
    % Getting all trials for each patient
    trailList = trailList([trailList.isdir] & ~ismember({trailList.name}, {'.', '..'}));
    

    %% Looping over trials
    for j = 1:length(trailList)
        trialName = trailList(j).name;
        trialPath = fullfile(folderPath, trialName);
        trialID = matlab.lang.makeValidName(['T' num2str(j)]);

        ForceShoeData = loadMTxDataDir(trialPath, 1);
        

        %% Offsetting Forces
        for forceSensorCount = 1:length(ForceShoeData)
            % Moving the original measurement to different name in struct
            ForceShoeData{forceSensorCount}.F_withoutoffset = ForceShoeData{forceSensorCount}.F;
            
            % Replacing F by offsetted force measurements
            offsetForce = mean(IFSoffset{forceSensorCount}.F, 2);
            ForceShoeData{forceSensorCount}.F = ForceShoeData{forceSensorCount}.F_withoutoffset - offsetForce;
        end
        

        %% Initialising arrays
        % Rotation to Shoe Frame: X=-X Y=Y and Z=-Z
        Rfs = [
            1 0 0;
            0 -1 0;
            0 0 1];
        
        N = ForceShoeData{1}.N;  % Number of points
        samplingFs = ForceShoeData{1}.fs;  % Sampling frequency

        Flf = zeros(3, N);  % Force left foot
        Flh = zeros(3, N);  % Force left heel
        Frf = zeros(3, N);  % Force right foot
        Frh = zeros(3, N);  % Force right heel

        arf = zeros(3, N);  % Acceleration right foot, needed for syncing
        
        %% Rotate forces and torques
        for sample = 1:N
            Flf(:,sample) = Rfs*ForceShoeData{1}.F(:,sample);
            Flh(:,sample) = Rfs*ForceShoeData{2}.F(:,sample);
            Frh(:,sample) = Rfs*ForceShoeData{3}.F(:,sample);
            Frf(:,sample) = Rfs*ForceShoeData{4}.F(:,sample);
            
            % Right Foot Accelerations

            arf(:, sample) = 0.5 * (ForceShoeData{3}.acc(:, sample) + ForceShoeData{4}.acc(:, sample));

        end


        %% Applying Butterworth Filter
        Flf = butterfilterlpf(Flf',10,samplingFs,2);
        Flh = butterfilterlpf(Flh',10,samplingFs,2);
        Frh = butterfilterlpf(Frh',10,samplingFs,2);
        Frf = butterfilterlpf(Frf',10,samplingFs,2);
        

        %% Extracting GRFs
        GRF_lf = Flh + Flf; % GRF of left foot
        GRF_rf = Frh + Frf; % GRF of right foot
        GRF_total = GRF_lf + GRF_rf;
        

        %% Storing GRF's in global struct
        GRFData.(patientID).(trialID) = struct( ...
            'Total', GRF_total, ...
            'Left', GRF_lf, ...
            'Right', GRF_rf);

        %% Storing right ankle accelerations (Needed for Syncing)
        RightFootData.(patientID).(trialID) = arf;


    end
end
%% Save to .mat files
save("C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Data Files\StraightWalking\MatLabCombined\GRF.mat", ...
    'GRFData')

save("C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Data Files\StraightWalking\MatLabCombined\IFS_Acc_RF.mat", ...
    'RightFootData')