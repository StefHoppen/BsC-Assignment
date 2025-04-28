clc
close all 
clear

CoM_struct = struct();

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

        comPos = (1 / 1e3) * (lasiPos + lpsiPos + rasiPos + rpsiPos) / 4; % m
        
        comX = comPos(:, 1);
        comY = comPos(:, 2);
        comZ = comPos(:, 3);
        
        trialData = struct( ...
            'x', comX', ...
            'y', comY', ...
            'z', comZ' ...
            );

        CoM_struct.(patientID).(trialID) = trialData;
    end
end
save("C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Data Files\StraightWalking\MatLabCombined\Vicon_CoM.mat", ...
    'CoM_struct')
