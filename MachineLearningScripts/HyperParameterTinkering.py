""" THIS SCRIPT AIMS TO TUNE THE HYPERPARAMETERS OF THE NAIVE MODEL. THE SAME HYPERPARAMETERS WILL BE USED FOR THE PINN
AND TESTED IF THE PINN PROVIDES ANY ADVANTAGE."""

from MachineLearningScripts.DataSets import StraightWalkingTCNSet
from MachineLearningScripts.MachineLearningClasses import TemporalConvolutionNetwork

import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

torch.manual_seed(0)  # For reproducibility
""" === TRAINING SETTINGS === """
EPOCHS = 100  # Number of times the entire dataset is trained on
LEARNING_RATE = 1e-3

""" === ARCHITECTURE SETTINGS === """
IN_CHANNELS = 24  # 4 IMUs with 6 channels
TCN_CHANNELS = [48, 48, 48, 48]  # Number of channels each TCN block has
TCN_DILATIONS = [1, 1, 2, 2]  # Dilation setting for each TCN block
KERNEL_SIZE = 2  # Size of the convolution kernel
P_DROPOUT = 0.3  # Chance of an input being zeroed out in the TCN block
OUTPUT_SIZE = 3 # X, Y, Z of COM

model = TemporalConvolutionNetwork(in_channels=IN_CHANNELS, # Initialise Model
                                   dilations=TCN_DILATIONS,
                                   channels=TCN_CHANNELS,
                                   kernel_size=KERNEL_SIZE,
                                   p_dropout=P_DROPOUT,
                                   outputs=OUTPUT_SIZE)
print(f'Receptive Field of Model {model.receptive_field} = {model.receptive_field*(1/100)}s')

""" === LOADING MISC === """
dataset = StraightWalkingTCNSet()
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
lossFunc = torch.nn.MSELoss()

""" === SPLITTING DATASET === """
trainSize = int(0.8*len(dataset))  # Figure out train/test size
trainSet, testSet = random_split(dataset, [trainSize, len(dataset)-trainSize])

# Batch sizes can only be one, since each step is of differnt length
trainLoader = DataLoader(trainSet, batch_size=1, shuffle=True)
testLoader = DataLoader(testSet, batch_size=1, shuffle=True)

""" === INITIALISING MONITORING PLOT === """
# THE PLOTTING IS ALL GENERATED CODE BY CHATGPT
plt.ion()
fig, axs = plt.subplots(3, 2, figsize=(10, 8), gridspec_kw={'width_ratios': [2, 3]})
fig.suptitle('Training Progress', fontsize=16)

# Flatten axes
loss_ax = axs[0, 0]
x_ax = axs[0, 1]
y_ax = axs[1, 1]
z_ax = axs[2, 1]

# Hide unused subplot slots
axs[1, 0].axis('off')
axs[2, 0].axis('off')

# Initialize line containers
trainLosses, = loss_ax.plot([], [], label='Train Loss')
testLosses, = loss_ax.plot([], [], label='Test Loss')
loss_ax.set_title('Loss over Epochs')
loss_ax.set_xlabel('Epoch')
loss_ax.set_ylabel('Loss')
loss_ax.legend()

xLine, = x_ax.plot([], [], label='X-AE')
yLine, = y_ax.plot([], [], label='Y-AE')
zLine, = z_ax.plot([], [], label='Z-AE')

x_ax.set_title('X Absolute Error (m)')
y_ax.set_title('Y Absolute Error (m)')
z_ax.set_title('Z Absolute Error (m)')
for ax in [x_ax, y_ax, z_ax]:
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Abs Error (m)')
    ax.legend()


""" === EPOCH LOOP === """
trainLossList, testLossList = [], []
xAEList, yAEList, zAEList = [], [], []
for i in range(EPOCHS):
    print(i)
    trainLoss, testLoss = 0, 0  # Initialise loss storage
    """ -- TRAINING -- """
    model.train()  # Put model in training mode
    for (patientID, trialID, stepID, IMU, COM, GRF), (comMean, comStd) in trainLoader:
        optim.zero_grad()  # Each batch gradients start at zero
        comHatNorm = model(IMU)  # Predict normalised COM position
        totalLoss = lossFunc(comHatNorm, COM)  # Computing loss
        totalLoss.backward()  # Compute gradients of parameters
        optim.step()  # Update model parameters
        trainLoss += totalLoss.item()  # Add loss to the tally
    trainLoss = trainLoss / len(trainLoader)  # Compute average loss over train set
    trainLossList.append(trainLoss)  # Store train loss history

    """ -- TESTING -- """
    model.eval()
    xAE, yAE, zAE = 0, 0, 0  # Create storages for absolute errors as well
    for (patientID, trialID, stepID, IMU, COM, GRF), (comMean, comStd) in testLoader:
        optim.zero_grad()  # Each batch gradients start at zero
        comHatNorm = model(IMU)  # Predict normalised COM position
        totalLoss = lossFunc(comHatNorm, COM)  # Computing loss
        testLoss += totalLoss.item()  # Add loss to the tally

        comHat = comHatNorm.squeeze()  * comStd + comMean
        COM = COM.squeeze() * comStd+ comMean

        testAE = torch.mean(torch.abs(COM - comHat).squeeze(), dim=1)
        xAE += testAE[0].item()
        yAE += testAE[1].item()
        zAE += testAE[2].item()

    testLoss = testLoss / len(testLoader)
    xAE = xAE / len(testLoader)
    yAE = yAE / len(testLoader)
    zAE = zAE / len(testLoader)

    testLossList.append(testLoss)
    xAEList.append(xAE)
    yAEList.append(yAE)
    zAEList.append(zAE)

    """ -- PLOTTING -- """
    epochs = list(range(len(trainLossList)))
    trainLosses.set_data(epochs, trainLossList)
    testLosses.set_data(epochs, testLossList)
    xLine.set_data(epochs, xAEList)
    yLine.set_data(epochs, yAEList)
    zLine.set_data(epochs, zAEList)

    for ax in [loss_ax, x_ax, y_ax, z_ax]:
        ax.relim()
        ax.autoscale_view()

    plt.draw()
    plt.pause(0.01)
plt.ioff()
plt.show()

