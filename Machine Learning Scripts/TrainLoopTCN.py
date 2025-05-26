from DataSets import StraightWalkingTCNSet
from MachineLearningClasses import TemporalConvolutionNetwork, CentralFDConv1D, PINNLoss, NaiveLoss

import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

""" TRAINING SETTINGS """
PINN = True

BATCH_SIZE = 1
EPOCHS = 1000
LEARNING_RATE = 1e-3

WPHYSICS = 0.4  # Multiplication factor of Physics loss

""" MODEL """
IN_CHANNELS = 24  # 4 IMUs, with 6 channels each
CHANNELS = [64, 64, 64, 64]
DILATIONS = [1, 2, 2, 4]
KERNEL_SIZE = 3
P_DROPOUT = 0.5
OUTPUT_SIZE = 3

model = TemporalConvolutionNetwork(in_channels=IN_CHANNELS, # Initialise Model
                                   dilations=DILATIONS,
                                   channels=CHANNELS,
                                   kernel_size=KERNEL_SIZE,
                                   p_dropout=P_DROPOUT,
                                   outputs=OUTPUT_SIZE)
pinnExtension = CentralFDConv1D(stencil_radius=3, h=0.1)

modelReceptField = model.receptive_field
print(f'Receptive Field of Model {modelReceptField} = {modelReceptField*(1/100)}s')

""" DATASET """
dataset = StraightWalkingTCNSet()  # Initialisation

trainSize = int(0.8*len(dataset))  # Figure out train/test size
trainSet, testSet = random_split(dataset, [trainSize, len(dataset)-trainSize])

trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True)
testLoader = DataLoader(testSet, batch_size=BATCH_SIZE, shuffle=True)

""" MISCELLANEOUS """
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

""" TRAINING LOOP """
for i in range(EPOCHS):
    trainLoss, testLoss = 0, 0  # Initialise storage for loss values

    """ TRAINING """
    model.train()  # Set model in train mode

    # Looping over the training set
    for (patientID, trialID, time, IMU, COM, GRF), (comMean, comStd) in trainLoader:
        optim.zero_grad()  # Each batch gradients start at zero

        comHat = model(IMU)  # Predict normalised COM position with network

        if PINN:
            comHatM = (comHat * comStd + comMean).float()  # Transferring prediction back to pre-normalised dimensions (meters)
            comAcc = pinnExtension(comHatM)
            physicsLoss = torch.mean((comAcc[:, :, modelReceptField:] - GRF[:, :, modelReceptField:])**2)
        else:
            physicsLoss = torch.tensor(0)
        naiveLoss = torch.mean((comHat[:, :, modelReceptField:] - COM[:, :, modelReceptField:])**2)

        totalLoss = naiveLoss + WPHYSICS*physicsLoss
        totalLoss.backward()  # Compute gradients wrt the model parameters
        optim.step()  # Update the model parameters

        trainLoss += totalLoss.item()  # Adding the loss to the storage
    trainLoss = trainLoss / len(trainLoader)  # Averaging the train loss over the epoch

    """ TESTING """
    model.eval()  # Set model in evaluation mode
    xError, yError, zError = 0, 0, 0

    # Loop over the test set
    for (patientID, trialID, time, IMU, COM, GRF), (comMean, comStd) in testLoader:
        comHat = model(IMU)  # Predict COM position with network

        #totalLoss = lossFunc(comHat[:, 1:, modelReceptField:], COM[:, 1:, modelReceptField:]) # Computing Naive Loss
        if PINN:
            comHatM = (comHat * comStd + comMean).float()  # Transferring prediction back to pre-normalised dimensions (meters)
            comAcc = pinnExtension(comHatM)
            physicsLoss = torch.mean((comAcc[:, :, modelReceptField:] - GRF[:, :, modelReceptField:]) ** 2)
        else:
            physicsLoss = torch.tensor(0)
        naiveLoss = torch.mean((comHat[:, :, modelReceptField:] - COM[:, :, modelReceptField:]) ** 2)

        totalLoss = naiveLoss + WPHYSICS * physicsLoss
        testLoss += totalLoss.item()

        normTestError = torch.mean(torch.abs(comHat[:, :, modelReceptField:] - COM[:, :, modelReceptField:]) , dim=2).squeeze()
        testError = normTestError * comStd.squeeze() + comMean.squeeze()

        xError += testError[0].item()
        yError += testError[1].item()
        zError += testError[2].item()

    testLoss = testLoss / len(testLoader)
    xError = xError / len(testLoader)
    yError = yError / len(testLoader)
    zError = zError / len(testLoader)

    print(f'Epoch {i + 1}/{EPOCHS} | Train Loss: {trainLoss:.6f} | Test Loss: {testLoss:.6f}\n'
          f'Absolute errors(m) X: {xError}, Y: {yError}, Z: {zError}\n'
          f'')

""" Plotting all test trials """

for (patientID, trialID, time, IMU, COM, GRF), comSTD in testLoader:
    comHat = model(IMU)  # Predict COM position with network

    comHatM = (comHat * comStd + comMean).float() # Transferring prediction back to pre-normalised dimensions (meters)
    comAcc = pinnExtension(comHatM)

    # Ensure shapes are correct
    comHat = comHat.squeeze(0).detach().cpu().numpy()  # [3, T]
    COM = COM.squeeze(0).detach().cpu().numpy()
    GRF = GRF.squeeze(0).detach().cpu().numpy()
    comAcc = comAcc.squeeze(0).detach().cpu().numpy()


 # Plot
    fig, axs = plt.subplots(6, 1, figsize=(10, 8), sharex=True)
    labels = ['x', 'y', 'z', 'Acc x', 'Acc y', 'Acc z']

    for i in range(3):
        axs[i].plot(COM[i], label='True', linewidth=2)
        axs[i].plot(comHat[i], label='Predicted', linestyle='--')
        axs[i].set_ylabel(f'COM {labels[i]}')
        axs[i].legend()
        axs[i].grid(True)
    for i in range(3, 6):
        axs[i].plot(GRF[i-3], label='True', linewidth=2)
        axs[i].plot(comAcc[i-3], label='Predicted', linestyle='--')
        axs[i].set_ylabel(f'COM {labels[i]}')
        axs[i].legend()
        axs[i].grid(True)

    axs[-1].set_xlabel('Time Step')
    fig.suptitle(f'Patient {patientID[0]} | Trial {trialID[0]}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


