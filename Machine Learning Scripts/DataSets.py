import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import itertools


class StraightWalkSet(Dataset):
    """ Straight Walking Dataset where an arbitrary window size can be set """
    accountingCols = ['Patient', 'Trial']
    def __init__(self, window_size: int):
        self.dataDirectory = r'C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Data Files\StraightWalking\Python'
        self.window_size = window_size

        # Loading IMU Data
        imuLF = pd.read_csv(os.path.join(self.dataDirectory, 'IMU_LF.csv'))
        imuRF = pd.read_csv(os.path.join(self.dataDirectory, 'IMU_LF.csv'))
        imuPEL = pd.read_csv(os.path.join(self.dataDirectory, 'IMU_Pel.csv'))
        imuSTE = pd.read_csv(os.path.join(self.dataDirectory, 'IMU_Ste.csv'))

        # Loading GRF Data
        GRF = pd.read_csv(os.path.join(self.dataDirectory, 'GRF.csv'))

        # Loading COM Position Data (VICON)
        COM = pd.read_csv(os.path.join(self.dataDirectory, 'COM.csv'))

        # Extracting Measurement Columns
        self.imuCols = [col for col in imuLF.columns if col not in self.accountingCols]
        self.grfCols = [col for col in GRF.columns if col not in self.accountingCols]
        self.comCols = [col for col in COM.columns if col not in self.accountingCols]

        # Converting to torch.Tensors
        self.imuLF = torch.tensor(imuLF[self.imuCols].to_numpy(), dtype=torch.float32)
        self.imuRF = torch.tensor(imuRF[self.imuCols].to_numpy(), dtype=torch.float32)
        self.imuPEL = torch.tensor(imuPEL[self.imuCols].to_numpy(), dtype=torch.float32)
        self.imuSTE = torch.tensor(imuSTE[self.imuCols].to_numpy(), dtype=torch.float32)

        self.GRF = torch.tensor(GRF[self.grfCols].to_numpy(), dtype=torch.float32)

        self.COM = torch.tensor(COM[self.comCols].to_numpy(), dtype=torch.float32)

    def __len__(self):
        return self.COM.shape[0] - self.window_size

    def __getitem__(self, idx):
        # Inputs are of window size
        idx += self.window_size  # Shift index a window size forward

        imuLF = self.imuLF[idx-self.window_size : idx]
        imuRF = self.imuRF[idx-self.window_size : idx]
        imuPEL = self.imuPEL[idx-self.window_size : idx]
        imuSTE = self.imuSTE[idx-self.window_size : idx]

        COM = self.COM[idx]
        GRF = self.GRF[idx]

        return (imuLF, imuRF, imuPEL, imuSTE), COM, GRF

class StraightWalkingTCNSet(Dataset):
    """ Straight Walking Dataset which is convenient for a TCN, it returns an entire trial at once. """
    def __init__(self):
        self.dataDirectory = r'C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Data Files\StraightWalking\Python'

        # Loading IMU Data
        imuLF = pd.read_csv(os.path.join(self.dataDirectory, 'IMU_LF.csv'))
        imuRF = pd.read_csv(os.path.join(self.dataDirectory, 'IMU_LF.csv'))
        imuPEL = pd.read_csv(os.path.join(self.dataDirectory, 'IMU_Pel.csv'))
        imuSTE = pd.read_csv(os.path.join(self.dataDirectory, 'IMU_Ste.csv'))

        GRF = pd.read_csv(os.path.join(self.dataDirectory, 'GRF.csv'))

        # Loading COM Position Data (VICON)
        COM = pd.read_csv(os.path.join(self.dataDirectory, 'COM.csv'))
        self.comMean = torch.tensor(COM[['Anteroposterior','Mediolateral', 'Vertical']].mean()).unsqueeze(dim=1)
        self.comStd = torch.tensor(COM[['Anteroposterior','Mediolateral', 'Vertical']].std()).unsqueeze(dim=1)

        # Retrieving Patient and Trial info
        patientSet = set(imuLF['Patient'].to_list())
        trialSet = set(imuLF['Trial'].to_list())

        self.samples= []
        for patientID, trialID in itertools.product(patientSet, trialSet):
            # In theory, all match indexes should be the same. But just to be cautious, I do this.
            matchIdxIMU = (imuLF['Patient'] == patientID) & (imuLF['Trial'] == trialID)
            matchIdxGRF = (GRF['Patient'] == patientID) & (GRF['Trial'] == trialID)
            matchIdxCOM = (COM['Patient'] == patientID) & (COM['Trial'] == trialID)


            time = torch.tensor(imuLF[matchIdxIMU].iloc[:, 6].to_numpy()).unsqueeze(dim=1).T.float()

            # IMU DATA
            LF = torch.tensor(imuLF[matchIdxIMU].iloc[:, :6].to_numpy())
            RF = torch.tensor(imuRF[matchIdxIMU].iloc[:, :6].to_numpy())
            PEL = torch.tensor(imuPEL[matchIdxIMU].iloc[:, :6].to_numpy())
            STE = torch.tensor(imuSTE[matchIdxIMU].iloc[:, :6].to_numpy())
            imuSample = torch.concat([LF, RF, PEL, STE], dim=1).T.float()

            # COM DATA
            comSample = ((torch.tensor(COM[matchIdxCOM].iloc[:, :3].to_numpy()).T - self.comMean) / self.comStd).float()

            # GRF DATA
            grfSample = torch.tensor(GRF[matchIdxGRF].iloc[:, :3].to_numpy()).T.float()

            # TOTAL SAMPLE
            trialSample = [patientID, trialID, time, imuSample, comSample, grfSample]

            self.samples.append(trialSample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], (self.comMean, self.comStd)
