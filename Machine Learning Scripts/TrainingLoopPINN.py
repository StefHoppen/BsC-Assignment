from MachineLearningClasses import *
from DataSets import *

import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

def computeDerivative(output: torch.tensor, time: torch.tensor):
    X = torch.autograd.grad(
        outputs= output[:, 0],
        inputs=time,
        grad_outputs=torch.ones_like(output[:, 0]),
        create_graph=True)[0]

    Y = torch.autograd.grad(
        outputs=output[:, 1],
        inputs=time,
        grad_outputs=torch.ones_like(output[:, 1]),
        create_graph=True)[0]

    Z = torch.autograd.grad(
        outputs=output[:, 2],
        inputs=time,
        grad_outputs=torch.ones_like(output[:, 2]),
        create_graph=True)[0]
    return torch.concat([X, Y, Z], dim=1)



# === Training Hyperparameters ===
window_size = 1
batch_size = 64
epochs = 300
learning_rate = 1e-3
naiveCoeff = 0.1
physicsCoeff = 10


print("PyTorch version:", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


""" Model Initialisation """
# Model Hyperparameters
model_architecture = [25, 64, 64, 32, 6]
activation_function = 'Tanh'
p_dropout = 0.1


model = FeedForwardNeuralNetwork(architecture=model_architecture, act_func=activation_function, p_dropout=p_dropout)
lossFuncNaive = torch.nn.MSELoss()
lossFuncPhysics = torch.nn.L1Loss()
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
        optim.zero_grad() # Starting with all gradients at zero
        t = IMU[0][:, -1, -1].unsqueeze(dim=1)  # Extracting time step from IMU data
        #t.requires_grad_(True)  # Enabling gradient on time tensor, for automatic differentiation
        IMU = torch.stack(IMU, dim=2)[:, :, :, :-1]  # Stack all IMU data on top of another
        IMU = torch.flatten(IMU, start_dim=1)  # Flatten the stacked data
        IMU = torch.concat([t, IMU], dim=1)  # Append time to the input

        pred = model(IMU)
        CoM_hat = pred[:, 0:3]
        CoM_acc = pred[:, 3:6]


        """
        CoM_hat = model(IMU)  # Making CoM prediction

        CoM_vel = computeDerivative(output=CoM_hat, time=t)  # Automatic differentiation

        CoM_acc = computeDerivative(output=CoM_vel, time=t)  # Automatic differentiation
        """
        naiveLoss = lossFuncNaive(CoM_hat, CoM)  # Computing naive loss
        physicsLoss = lossFuncPhysics(CoM_acc, GRF)  # L1 Loss

        totalLoss = naiveCoeff*naiveLoss + physicsCoeff*physicsLoss # Combining both loss terms

        totalLoss.backward()  # Calculating gradients with respect to the loss
        optim.step()  # Update the weights

        trainNaiveLoss += naiveLoss.item()*naiveCoeff  # Storing naive loss
        trainPhysicsLoss += physicsLoss.item()*physicsCoeff
        trainLoss += totalLoss.item()  # Adding loss to total loss for the epoch
    trainLoss = trainLoss / len(trainLoader)  # Dividing the total loss to get loss per sample
    trainNaiveLoss = trainNaiveLoss / len(trainLoader)
    trainPhysicsLoss = trainPhysicsLoss / len(trainLoader)


    # Testing
    model.eval()  # Set model in evaluation mode
    xError, yError, zError = 0, 0, 0
    for IMU, CoM, GRF in testLoader:
        t = IMU[0][:, -1, -1].unsqueeze(dim=1)  # Extracting time step from IMU data
        #t.requires_grad_(True)  # Enabling gradient on time tensor, for automatic differentiation
        IMU = torch.stack(IMU, dim=2)[:, :, :, :-1]  # Stack all IMU data on top of another
        IMU = torch.flatten(IMU, start_dim=1)  # Flatten the stacked data
        IMU = torch.concat([t, IMU], dim=1)  # Append time to the input

        pred = model(IMU)
        CoM_hat = pred[:, 0:3]
        CoM_acc = pred[:, 3:6]

        """
        CoM_hat = model(IMU)

        CoM_vel = computeDerivative(output=CoM_hat, time=t)  # Automatic differentiation

        CoM_acc = computeDerivative(output=CoM_vel, time=t)  # Automatic differentiation
        """

        naiveLoss = lossFuncNaive(CoM_hat, CoM)  # Computing naive loss
        physicsLoss = lossFuncPhysics(CoM_acc, GRF)   # L1 Loss

        totalLoss = naiveCoeff * naiveLoss + physicsCoeff * physicsLoss  # Combining both loss terms
        testLoss += totalLoss.item()

        testNaiveLoss += naiveLoss.item()*naiveCoeff
        testPhysicsLoss += physicsLoss.item()*physicsCoeff

        testError = torch.mean(torch.abs(CoM_hat - CoM), dim=0)
        xError += testError[0].item()
        yError += testError[1].item()
        zError += testError[2].item()

    testLoss = testLoss / len(testLoader)
    testNaiveLoss = testNaiveLoss / len(testLoader)
    testPhysicsLoss = testPhysicsLoss / len(testLoader)

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
          f"Physics Losses | Train: {trainPhysicsLoss:.6f} | Test: {testPhysicsLoss:.6f}\n"
          f"Naive Losses | Train: {trainNaiveLoss:.6f} | Test: {testNaiveLoss:.6f}\n"
          f"Absolute errors (cm) X: {xError}, Y: {yError}, Z: {zError}\n"
          f"")

#plt.ioff()  # Turn off interactive mode
#plt.show()  # Final display if you want a clean static plot at the end