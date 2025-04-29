%% Load data

clear all; close all
% load Forceshoe
addpath(genpath('Functions'))

%% initialize variables
filename_IFS = 'C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Data Files\StraightWalking\MatLab\ForceShoe\191008\191008_T01';

dataOrderForceShoes = [4; 1]; % RF, LF
%%
ForceShoeData = loadMTxDataDir(filename_IFS,1);% numsamples as last dim!


%% ForceShoe offset values
[IFS_directory,IFS_name,~]=fileparts(filename_IFS);

    load([IFS_directory '\191008_offset.mat']);


%% use offsets to estimate Forces

for forceSensorCount = 1:length(ForceShoeData)
    ForceShoeData{forceSensorCount}.F_withoutoffset = ForceShoeData{forceSensorCount}.F;
    
    % using offsets
    offsetForce = mean(IFSoffsets{forceSensorCount}.F,2);    

    ForceShoeData{forceSensorCount}.F = ForceShoeData{forceSensorCount}.F_withoutoffset - offsetForce;
    
end
%% Initialisation

% Rotation force sensor to shoe frame X=-X Y=Y en Z=-Z

Rfs = [1 0  0;
    0 -1  0;
    0 0 1];

N = ForceShoeData{1}.N;
samplingFs = ForceShoeData{1}.fs;
Flf = zeros(3,N);
Flh = zeros(3,N);
Frh = zeros(3,N);
Frf = zeros(3,N);

%% Rotate forces and torques
for sample = 1 : N
    Flf(:,sample) = Rfs*ForceShoeData{1}.F(:,sample);
    Flh(:,sample) = Rfs*ForceShoeData{2}.F(:,sample);
    Frh(:,sample) = Rfs*ForceShoeData{3}.F(:,sample);
    Frf(:,sample) = Rfs*ForceShoeData{4}.F(:,sample);
   
end

% Filter forces, and translate to (N x 3)
Flf = butterfilterlpf(Flf',10,samplingFs,2);
Flh = butterfilterlpf(Flh',10,samplingFs,2);
Frh = butterfilterlpf(Frh',10,samplingFs,2);
Frf = butterfilterlpf(Frf',10,samplingFs,2);

rightFoot = Frh + Frf;
leftFoot = Flf + Flh;
totalGRF = rightFoot + leftFoot;