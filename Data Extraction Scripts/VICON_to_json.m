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
    patientID = matlab.lang.makeValidName(['P_' folderName]);
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

% Encode and save
jsonStr = jsonencode(CoM_struct, 'PrettyPrint', true);  % Nice formatting
fid = fopen('VICON_COM.json', 'w');
if fid == -1
    error('Cannot create JSON file');
end
fwrite(fid, jsonStr, 'char');
fclose(fid);
