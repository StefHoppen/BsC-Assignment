from MachineLearningScripts.MachineLearningClasses import *
import os
import torch

""" === ARCHITECTURE SETTINGS === """
IN_CHANNELS = 25  # 4 IMUs with 6 channels + time
TCN_CHANNELS = [48, 48, 48, 48]  # Number of channels each TCN block has
TCN_DILATIONS = [1, 1, 2, 2]  # Dilation setting for each TCN block
KERNEL_SIZE = 2  # Size of the convolution kernel
P_DROPOUT = 0.3  # Chance of an input being zeroed out in the TCN block
OUTPUT_SIZE = 3  # X, Y, Z of COM
LEARNING_RATE = 1e-4

""" PINN SETTINGS """
STENCIL_RADIUS = 3
F_SAMPLE = 100
POLY_DEG = 10

class TcnFinal:
    def __init__(self):
        self.model = TemporalConvolutionNetwork(
            in_channels=IN_CHANNELS,
            dilations=TCN_DILATIONS,
            channels=TCN_CHANNELS,
            kernel_size=KERNEL_SIZE,
            p_dropout=P_DROPOUT,
            outputs=OUTPUT_SIZE)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.lossFunc = torch.nn.MSELoss()

        self.saveDir = r'C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\MachineLearningScripts\ModelParameters'

    def trainEpoch(self, dataLoader: torch.utils.data.DataLoader):
        trainLoss = 0  # Initialize loss as zero
        nBatch = 0
        self.model.train()  # Put model in training mode (activates dropout etc.)
        for (patientID, trialID, stepID, IMU, COM, GRF), (comMean, comStd) in dataLoader:
            nBatch += 1
            self.optim.zero_grad()  # Set all gradients to zero eatch batch
            comHat = self.model(IMU)  # Make normalised prediction

            totalLoss = self.lossFunc(comHat, COM)  # Computing loss
            totalLoss.backward()  # Compute gradients of parameters
            self.optim.step()  # Update model parameters
            trainLoss += totalLoss.item()  # Add loss to the tally

        lossDict = {  # This is just for consistency in testing at the end
            'Weighted': trainLoss / nBatch,
            'Naive': trainLoss / nBatch,
            'Informed': 0}


        return lossDict  # Return the trainLoss

    def testEpoch(self, dataLoader: torch.utils.data.DataLoader):
        testLoss = 0  # Initialize loss as zero
        self.model.eval()  # Set model in testing mode
        perfDict = {
            'Loss': None,
            'MAE': {},
            'RMAE': {}
        }
        xMAE, yMAE, zMAE = 0, 0, 0  # Initialising variables for Mean Absolute Errors

        nBatch = 0
        for (patientID, trialID, stepID, IMU, COM, GRF), (comMean, comStd) in dataLoader:
            nBatch += 1
            comHat = self.model(IMU) # Predict normalised COM position using network
            totalLoss = self.lossFunc(comHat, COM)  # Computing loss
            testLoss += totalLoss.item()

            # Assessing Performance
            COM = COM.squeeze() * comStd + comMean  # Converting reference signal to meter
            comHat = comHat.squeeze() * comStd + comMean  # Converting prediction signal to meter

            testMAE = torch.mean(torch.abs(COM - comHat).squeeze(), dim=1)  # Computing absolute error

            # Storing Mean Absolute Errors
            xMAE += testMAE[0].item()
            yMAE += testMAE[1].item()
            zMAE += testMAE[2].item()


        # Computing averages over set
        lossDict = {
            'Weighted': testLoss / nBatch,
            'Naive': testLoss / nBatch,
            'Informed': 0}

        perfDict = {
            'MAE': {
                'X': xMAE / nBatch,
                'Y': yMAE / nBatch,
                'Z': zMAE / nBatch}}
        return lossDict, perfDict

    def saveParams(self, foldIdx):
        fileName = f'{foldIdx}TCN.pth'
        filePath = os.path.join(self.saveDir, fileName)
        torch.save({'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optim.state_dict()},
                   filePath)
        print(f'Model is save to {filePath}')

    def loadParams(self, foldIdx):
        fileName = f'{foldIdx}TCN.pth'
        filePath = os.path.join(self.saveDir, fileName)
        checkpoint = torch.load(filePath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {filePath}")

    def predict(self, x: torch.tensor):
        return self.model(x)

class PinnFinal:
    def __init__(self, WPINN):
        self.model = TemporalConvolutionNetwork(
            in_channels=IN_CHANNELS,
            dilations=TCN_DILATIONS,
            channels=TCN_CHANNELS,
            kernel_size=KERNEL_SIZE,
            p_dropout=P_DROPOUT,
            outputs=OUTPUT_SIZE)
        self.WPINN = WPINN
        self.differentiator = SimpleFDConv1D(stencil_radius=3, h=1/F_SAMPLE)

        self.optim = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.naiveLossFunc = torch.nn.MSELoss()
        self.informedLossFunc = torch.nn.L1Loss()

        self.saveDir = r'C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\MachineLearningScripts\ModelParameters'

    def trainEpoch(self, dataLoader: torch.utils.data.DataLoader):
        trainLoss = 0  # Initialize loss as zero
        naiveEpochLoss, informedEpochLoss = 0, 0

        self.model.train()  # Put model in training mode (activates dropout etc.)

        nBatch = 0
        for (patientID, trialID, stepID, IMU, COM, GRF), (comMean, comStd) in dataLoader:
            nBatch += 1
            self.optim.zero_grad()  # Set all gradients to zero each batch
            comHat = self.model(IMU)  # Make normalised prediction
            comHatSmoothed = polyFit(comHat, poly_deg=POLY_DEG, f_sample=F_SAMPLE)  # Smooth prediction
            comHatMeter = comHatSmoothed * comStd + comMean

            comHatAcc = self.differentiator(comHatMeter)  # Calculate acceleration
            comHatAcc[:, 2, :] = comHatAcc[:, 2, :] + 9.81  # Add gravitational constant
            comAccMag = torch.linalg.vector_norm(comHatAcc, dim=1)  # Compute the magnitude of the vector

            GRF = GRF[:, :, STENCIL_RADIUS: -STENCIL_RADIUS]  # Trim the edges
            GRFMag = torch.linalg.vector_norm(GRF, dim=1)  # Compute magnitude of GRF

            # Computing Losses
            naiveLoss = self.naiveLossFunc(comHat, COM)
            informedLoss = self.informedLossFunc(comAccMag, GRFMag)
            totalLoss = naiveLoss + self.WPINN*informedLoss

            # Backpropagation
            totalLoss.backward()  # Compute gradients of parameters
            self.optim.step()  # Update model parameters

            # Storing Losses
            trainLoss += totalLoss.item()
            naiveEpochLoss += naiveLoss.item()
            informedEpochLoss += informedLoss.item()

        # Computing average losses over the epoch
        lossDict = {
            'Weighted': trainLoss / nBatch,
            'Naive': naiveEpochLoss / nBatch,
            'Informed': informedEpochLoss / nBatch}
        return lossDict # Return the trainLoss

    def testEpoch(self, dataLoader: torch.utils.data.DataLoader):
        testLoss = 0  # Initialize loss as zero
        naiveEpochLoss, informedEpochLoss = 0, 0

        self.model.eval()  # Set model in testing mode
        xMAE, yMAE, zMAE = 0, 0, 0  # Initialising variables for Mean Absolute Errors

        nBatch = 0
        for (patientID, trialID, stepID, IMU, COM, GRF), (comMean, comStd) in dataLoader:
            nBatch += 1
            comHat = self.model(IMU)  # Make normalised prediction
            comHatSmoothed = polyFit(comHat, poly_deg=POLY_DEG, f_sample=F_SAMPLE)  # Smooth prediction
            comHatMeter = comHatSmoothed * comStd + comMean  # Compute back to meter

            comHatAcc = self.differentiator(comHatMeter)  # Calculate acceleration
            comHatAcc[:, 2, :] = comHatAcc[:, 2, :] + 9.81  # Add gravitational constant
            comAccMag = torch.linalg.vector_norm(comHatAcc, dim=1)  # Compute the magnitude of the vector

            GRF = GRF[:, :, STENCIL_RADIUS: -STENCIL_RADIUS]  # Trim the edges
            GRFMag = torch.linalg.vector_norm(GRF, dim=1)  # Compute magnitude of GRF

            # Computing Losses
            naiveLoss = self.naiveLossFunc(comHat, COM)
            informedLoss = self.informedLossFunc(comAccMag, GRFMag)
            totalLoss = naiveLoss + self.WPINN * informedLoss

            # Storing Losses
            testLoss += totalLoss.item()
            naiveEpochLoss += naiveLoss.item()
            informedEpochLoss += informedLoss.item()

            comMeter = COM.squeeze() * comStd + comMean  # Converting reference signal to meter

            # Assessing performance
            testMAE = torch.mean(torch.abs(comMeter - comHatMeter).squeeze(), dim=1)  # Computing absolute error

            # Storing Mean Absolute Errors
            xMAE += testMAE[0].item()
            yMAE += testMAE[1].item()
            zMAE += testMAE[2].item()

        # Computing averages over set
        lossDict = {
            'Weighted': testLoss / nBatch,
            'Naive': naiveEpochLoss / nBatch,
            'Informed': informedEpochLoss / nBatch}

        perfDict = {
            'MAE': {
                'X': xMAE / nBatch,
                'Y': yMAE / nBatch,
                'Z': zMAE / nBatch}}
        return lossDict, perfDict

    def saveParams(self, foldIdx):
        wPinnString = str(self.WPINN).replace('.', '')
        fileName = f'{foldIdx}PINN{wPinnString}.pth'
        filePath = os.path.join(self.saveDir, fileName)
        torch.save({'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optim.state_dict()},
                   filePath)
        print(f'Model is save to {filePath}')

    def loadParams(self, foldIdx):
        wPinnString = str(self.WPINN).replace('.', '')
        fileName = f'{foldIdx}PINN{wPinnString}.pth'
        filePath = os.path.join(self.saveDir, fileName)
        checkpoint = torch.load(filePath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {filePath}")

    def predict(self, x:torch.tensor):
        return self.model(x)