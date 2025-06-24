import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import itertools

class StraightWalkingTCNSet(Dataset):
    """ Straight Walking Dataset which is convenient for a TCN, it returns a complete step at once. """
    def __init__(self, device='cpu'):
        self.device = device
        self.dataDirectory = r'C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Data Files\StraightWalking\Python'

        # Loading IMU Data
        imuLF = pd.read_csv(os.path.join(self.dataDirectory, 'IMU_LF.csv'))
        imuRF = pd.read_csv(os.path.join(self.dataDirectory, 'IMU_LF.csv'))
        imuPEL = pd.read_csv(os.path.join(self.dataDirectory, 'IMU_Pel.csv'))
        imuSTE = pd.read_csv(os.path.join(self.dataDirectory, 'IMU_Ste.csv'))
        imuTime = pd.read_csv(os.path.join(self.dataDirectory, 'IMU_Time.csv'))  # THIS IS A TEST

        GRF = pd.read_csv(os.path.join(self.dataDirectory, 'GRF.csv'))

        # Loading COM Position Data (VICON)
        COM = pd.read_csv(os.path.join(self.dataDirectory, 'COM.csv'))
        self.comMean = torch.tensor(COM[['x', 'y', 'z']].mean(), dtype=torch.float32).unsqueeze(dim=1)
        self.comStd = torch.tensor(COM[['x', 'y', 'z']].std(), dtype=torch.float32).unsqueeze(dim=1)

        # Retrieving Patient and Trial info
        patientSet = sorted(set(imuLF['Patient'].to_list()))
        trialSet = sorted(set(imuLF['Trial'].to_list()))
        stepSet = sorted(set(imuLF['Step'].tolist()))

        self.samples= []
        for patientID, trialID, stepID in itertools.product(patientSet, trialSet, stepSet):
            # In theory, all match indexes should be the same. But just to be cautious, I do this.
            matchIdxIMU = (imuLF['Patient'] == patientID) & (imuLF['Trial'] == trialID) & (imuLF['Step'] == stepID)
            matchIdxGRF = (GRF['Patient'] == patientID) & (GRF['Trial'] == trialID) & (GRF['Step'] == stepID)
            matchIdxCOM = (COM['Patient'] == patientID) & (COM['Trial'] == trialID) & (COM['Step'] == stepID)

            if not matchIdxIMU.sum() == 0:  # Some trials contain more steps, thus some combinations of Patient, Trial,
                # Step result in no matching indexes. These are now skipped

                # IMU DATA
                LF = torch.tensor(imuLF[matchIdxIMU].iloc[:, :6].to_numpy())
                RF = torch.tensor(imuRF[matchIdxIMU].iloc[:, :6].to_numpy())
                PEL = torch.tensor(imuPEL[matchIdxIMU].iloc[:, :6].to_numpy())
                STE = torch.tensor(imuSTE[matchIdxIMU].iloc[:, :6].to_numpy())

                TIME = torch.tensor(imuTime[matchIdxIMU].iloc[:, :6].to_numpy())

                imuSample = torch.concat([LF, RF, PEL, STE, TIME], dim=1).T.float()

                # COM DATA
                comSample = ((torch.tensor(COM[matchIdxCOM].iloc[:, :3].to_numpy()).T - self.comMean) / self.comStd).float()

                # GRF DATA
                grfSample = torch.tensor(GRF[matchIdxGRF].iloc[:, :3].to_numpy()).T.float()

                # TOTAL SAMPLE
                stepSample = [patientID, trialID, stepID, imuSample, comSample, grfSample]

                self.samples.append(stepSample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], (self.comMean, self.comStd)
