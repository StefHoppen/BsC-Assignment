function data = loadMTxDataDir(folder_name,numSamplesAsLastDim)
% Load all MT-manager ASCI (tab-delimited) .txt files from one directory.
%
% Outputs:              a 'data' cell array (one cell for each sensor).
%
% (Optional) inputs:    -"folder_name" a folder from which the files are
%                        loaded. If not specified, uigetdir is used
%                       -"numSamplesAsLastDim" option to put numsamples as
%                        last dimension ([0],1)
%
% Examples:             data=loadMTwDataDir;
%                       data=loadMTwDataDir('D:\Dropbox\Metingen\20120425');
%                       data=loadMTwDataDir('D:\Dropbox\Metingen\20120425',1);
%
% Dirk Weenk, 2012-03-09
% Henk Kortier, 2013-02-27
% Dirk Weenk, 2013-03-08: correction for last missing samples
% Dirk Weenk, 2013-03-12: added option to put numsamples as last dimension ([0],1)
% Dirk Weenk, 2013-03-21: changed to read MTx files, exported with MT manager ForceShoe

if nargin<1
    % Select a dir
    startdir='../Measurements';
    folder_name = uigetdir(startdir);
elseif nargin<2
    numSamplesAsLastDim=0;
end

txtfiles=dir([folder_name '/*.txt']);

% rawfiles=dir([folder_name '/raw/*.mtb']);
% tmpfile = 'D:\Dropbox\Shared\Practicum BMT K8\Measurements\20130308\1\MTW\raw\MT_00200118-000.mtb';
% rawfile = readMtw(tmpfile,'xsb',[folder_name '/../../../../Matlab/']);

Nfiles = length(txtfiles);

if Nfiles == 0
    disp('no valid MTx files in this directory!');
    data = 0;
    return;
end
data=cell(1,Nfiles);
for i=1:Nfiles
    filename = txtfiles(i).name;
    fprintf('Start reading MTx file: %s\n',filename);
    data{i} = readCalibratedFSMTxLog([folder_name '/' filename]);
    data{i}.filename = filename;
end
% check and remove last samples
lastsamples=zeros(Nfiles,1);
for i=1:Nfiles
    lastsamples(i)=data{i}.counter(end);
end
endSampleCounter=min(lastsamples);
for i=1:Nfiles
    endSample=find(data{i}.counter==endSampleCounter);
    oldNumSamples=data{i}.N;
    % remove last samples only if oldNumSamples~=endSample
    if oldNumSamples~=endSample
        % store new number of samples
        data{i}.N=endSample;
        % loop over fieldnames
        fnames=fieldnames(data{i});
        for fn=1:length(fnames)
            % change only for the fields that contain arrays
            if size(data{i}.(fnames{fn}),1)==oldNumSamples
                % check if 2D array
                if ismatrix(data{i}.(fnames{fn}))
                    % store old and new data
                    olddata=data{i}.(fnames{fn});
                    newdata=olddata(1:endSample,:);
                    % clear old field values
                    data{i} = rmfield(data{i},fnames{fn});
                    % store new field values
                    data{i}.(fnames{fn})=newdata;
                elseif ndims(data{i}.(fnames{fn}))==3
                    % store old and new data
                    olddata=data{i}.(fnames{fn});
                    newdata=olddata(1:endSample,:,:);
                    % clear old field values
                    data{i} = rmfield(data{i},fnames{fn});
                    % store new field values
                    data{i}.(fnames{fn})=newdata;
                else
                    disp('Error: Ndims>3')
                end
            end
        end
    end
end
% double check if counters are the same, display message if not
counters=zeros(endSample,Nfiles);
for i=1:Nfiles
    counters(:,i)=data{i}.counter;
end
for i=2:Nfiles
    if ~isempty(find((counters(:,1)==counters(:,i))-1, 1))
        disp(['ERROR: Sample counters are different!'...
            ' There is a difference between MTW 1 and ' num2str(i) ...
            ' on sample #' num2str(find((counters(:,1)==counters(:,i))-1, 1)) ...
            '. But possibly on more sensors and samples.'])
    end
end
%% if numSamplesAsLastDim==1 then permute arrays
if numSamplesAsLastDim
    for i=1:Nfiles
        % loop over fieldnames
        fnames=fieldnames(data{i});
        for fn=1:length(fnames)
            % if char, do not permute/transpose (eg filename)
            if ischar(data{i}.(fnames{fn}))
                %
                % if scalar, do not permute/transpose (eg fs)
            elseif isscalar(data{i}.(fnames{fn}))
                %
                % if cell, do not permute/transpose (eg header)
            elseif iscell(data{i}.(fnames{fn}))
                %
                % check if 2D array, then transpose
            elseif ismatrix(data{i}.(fnames{fn}))
                % store old and new data
                olddata=data{i}.(fnames{fn});
                newdata=olddata';% Transpose to get numsamples as last dimension
                % clear old field values
                data{i} = rmfield(data{i},fnames{fn});
                % store new field values
                data{i}.(fnames{fn})=newdata;
                % if 3D array permute instead of transpose
            elseif ndims(data{i}.(fnames{fn}))==3
                % store old and new data
                olddata=data{i}.(fnames{fn});
                newdata=permute(olddata,[2 3 1]);% permute to get numsamples as last dimension
                % clear old field values
                data{i} = rmfield(data{i},fnames{fn});
                % store new field values
                data{i}.(fnames{fn})=newdata;
                % if >3D array
            else
                disp('Error: Ndims>3')
            end
        end
    end
end