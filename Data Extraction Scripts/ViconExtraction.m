clc
close all 
clear
%% Extracts the Left Foot, Right Foot and COM position of each trial
ViconData = struct();

parentDir = 'C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Data Files\StraightWalking\MatLab\VICON';

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
        rawdata = load(fullMatPath).rawdata;

        % Extracting CoM information
        lasiPos = rawdata.MarkerPos.LASI;
        lpsiPos = rawdata.MarkerPos.LPSI;

        rasiPos = rawdata.MarkerPos.RASI;
        rpsiPos = rawdata.MarkerPos.RPSI;

        comPos = (lasiPos + lpsiPos + rasiPos + rpsiPos) / 4; % m

        rfPos = rawdata.MarkerPos.RHEE;
        lfPos = rawdata.MarkerPos.LHEE;
        
        % Rotating coordinate frame such that the walking direction is in
        % positive x-direction. (If you switch x, then you must also switch
        % y)
        comPos(:, 1) = -comPos(:, 1);
        comPos(:, 2) = -comPos(:, 2);

        rfPos(:, 1) = -rfPos(:, 1);
        rfPos(:, 2) = -rfPos(:, 2);

        lfPos(:, 1) = -lfPos(:, 1);
        lfPos(:, 2) = -lfPos(:, 2);

        ViconData.(patientID).(trialID) = struct( ...
            'LF', lfPos, ...
            'RF', rfPos, ...
            'COM', comPos);

    end
end
save("C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Data Files\StraightWalking\MatLabCombined\Vicon.mat", ...
    'ViconData')
