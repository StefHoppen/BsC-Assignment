from DataSets import *
from MachineLearningClasses import FeedForwardNeuralNetwork

import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# === Training Hyperparameters ===
window_size = 1
batch_size = 64
epochs = 300
learning_rate = 1e-3

print("PyTorch version:", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

""" Model Initialisation """
# Model Hyperparameters
model_architecture = [25, 64, 64, 32, 3]
activation_function = 'Tanh'
p_dropout = 0.1

model = FeedForwardNeuralNetwork(architecture=model_architecture, act_func=activation_function, p_dropout=p_dropout)
lossFunc = torch.nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

""" Dataset Initialisation """
dataset = StraightWalkSet(window_size=window_size)

""" Splitting Dataset"""
trainSize = int(0.8 * len(dataset))
testSize = len(dataset) - trainSize
trainSet, testSet = random_split(dataset, [trainSize, testSize])

""" Initialising DataLoaders"""
trainLoader = DataLoader(trainSet, batch_size=batch_size, shuffle=True)
testLoader = DataLoader(testSet, batch_size=batch_size, shuffle=False)

""" Initialising Loss Plot """
"""
plt.ion()
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True)
trainNaiveLosses, testNaiveLosses = [], []
trainPhysicsLosses, testPhysicsLosses = [], []
trainLosses, testLosses = [], []
# Total Losses
trainLine, = ax1.plot([], [], label='Train Loss', color='blue')
testLine, = ax1.plot([], [], label='Test Loss', color='orange')
ax1.legend()

# Train Losses
trainNaiveLine, = ax2.plot([], [], label='Naive')
trainPhysicsLine, = ax2.plot([], [], label='Physics')
ax2.legend()

# Test Losses
testNaiveLine, = ax3.plot([], [], label='Naive')
testPhysicsLine, = ax3.plot([], [], label='Physics')
ax2.legend()

ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss')

"""

""" Training Loop """
for i in range(epochs):
    trainNaiveLoss, trainPhysicsLoss, testNaiveLoss, testPhysicsLoss = 0, 0, 0, 0
    trainLoss, testLoss = 0, 0
    # Training
    model.train()  # Setting the model in training mode (enables gradient calculation)

    # Looping over the training set
    for IMU, CoM, GRF in trainLoader:
        optim.zero_grad()  # Starting with all gradients at zero
        t = IMU[0][:, -1, -1].unsqueeze(dim=1)  # Extracting time step from IMU data
        IMU = torch.stack(IMU, dim=2)[:, :, :, :-1]  # Stack all IMU data on top of another
        IMU = torch.flatten(IMU, start_dim=1)  # Flatten the stacked data
        IMU = torch.concat([t, IMU], dim=1)  # Append time to the input

        CoM_hat = model(IMU)  # Making CoM prediction

        totalLoss = lossFunc(CoM_hat, CoM) *0.1 # Computing naive loss

        totalLoss.backward()  # Calculating gradients with respect to the loss
        optim.step()  # Update the weights

        trainLoss += totalLoss.item()  # Adding loss to total loss for the epoch
    trainLoss = trainLoss / len(trainLoader)  # Dividing the total loss to get loss per sample

    # Testing
    model.eval()  # Set model in evaluation mode
    xError, yError, zError = 0, 0, 0
    for IMU, CoM, GRF in testLoader:
        t = IMU[0][:, -1, -1].unsqueeze(dim=1)  # Extracting time step from IMU data
        IMU = torch.stack(IMU, dim=2)[:, :, :, :-1]  # Stack all IMU data on top of another
        IMU = torch.flatten(IMU, start_dim=1)  # Flatten the stacked data
        IMU = torch.concat([t, IMU], dim=1)  # Append time to the input

        CoM_hat = model(IMU)

        totalLoss = lossFunc(CoM_hat, CoM)*0.1  # Computing naive loss
        testLoss += totalLoss.item()

        testError = torch.mean(torch.abs(CoM_hat - CoM), dim=0)
        xError += testError[0].item()
        yError += testError[1].item()
        zError += testError[2].item()

    testLoss = testLoss / len(testLoader)

    xError = xError / len(testLoader)
    yError = yError / len(testLoader)
    zError = zError / len(testLoader)

    """
    # Storing the total losses for plotting
    trainLosses.append(trainLoss)
    testLosses.append(testLoss)

    # Storing physics and naive losses
    trainNaiveLosses.append(trainNaiveLoss)
    trainPhysicsLosses.append(trainPhysicsLoss)
    testNaiveLosses.append(testNaiveLoss)
    testPhysicsLosses.append(testPhysicsLoss)


    # Plotting
    # Total Loss
    trainLine.set_data(range(len(trainLosses)), trainLosses) # Update train loss data
    testLine.set_data(range(len(testLosses)), testLosses) # Update test loss data

    # Train loss
    trainNaiveLine.set_data(range(len(trainNaiveLosses)), trainNaiveLosses)
    trainPhysicsLine.set_data(range(len(trainPhysicsLosses)), trainPhysicsLosses)

    # Test loss
    testNaiveLine.set_data(range(len(testNaiveLosses)), testNaiveLosses)
    testPhysicsLine.set_data(range(len(testPhysicsLosses)), testPhysicsLosses)

    # Rescale all subplots
    for ax in (ax1, ax2, ax3):
        ax.relim()
        ax.autoscale_view()


    plt.draw() # Redraw the plot
    plt.pause(0.01)  # Pause to allow refresh

    """
    print(f"Epoch {i + 1}/{epochs} | Train Loss: {trainLoss:.6f} | Test Loss: {testLoss:.6f}\n"
          f"Absolute errors(cm) X: {xError}, Y: {yError}, Z: {zError}\n"
          f"")

# plt.ioff()  # Turn off interactive mode
# plt.show()  # Final display if you want a clean static plot at the end