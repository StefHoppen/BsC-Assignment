import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from BaseClasses import StraightWalkingDataset, FNN
import matplotlib.pyplot as plt

# === Hyperparameters ===
input_size = 24
output_size = 3
hidden_layers = 4
shape = 'Flat'
activation = 'ReLU'
batch_size = 64
epochs = 300
learning_rate = 1e-3

print("PyTorch version:", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

""" Model Initialisation """
model = FNN(input_size=input_size, n_hidden=hidden_layers, output_size=output_size, shape=shape, act_func=activation).to(device)
lossFunc = nn.MSELoss()
optim = optim.Adam(model.parameters(), lr=learning_rate)

""" Dataset Split """
dataset = StraightWalkingDataset(dev=device)

trainSize = int(0.8 * len(dataset))
testSize = len(dataset) - trainSize
trainSet, testSet = random_split(dataset, [trainSize, testSize])

trainLoader = DataLoader(trainSet, batch_size=batch_size, shuffle=True)
testLoader = DataLoader(testSet, batch_size=batch_size, shuffle=False)

""" Initialisng the plot """
plt.ion()
fig, ax = plt.subplots()
trainLosses, testLosses = [], []
trainLine, = ax.plot([], [], label='Train Loss', color='blue')
testLine, = ax.plot([], [], label='Test Loss', color='orange')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()


""" Training Loop """
for epoch in range(epochs):

    """ Training """
    # Put model in training mode
    model.train()
    # Storing training loss
    totalTrainLoss = 0

    # Looping over training batches
    for X, Y, _ in trainLoader:
        X, Y = X.float(), Y.float()

        # Setting all gradients to zero
        optim.zero_grad()

        # Calculating output with the network
        yHat = model(X)

        # Calculating loss
        loss = lossFunc(yHat, Y)

        # Backpropagating loss
        loss.backward()
        optim.step()
        totalTrainLoss += loss.item()

    trainLoss = totalTrainLoss / len(trainLoader)
    trainLosses.append(trainLoss)

    """ Testing """
    # Set model in evaluation mode
    model.eval()

    totalTestLoss = 0
    # Disabling gradient calculation (more efficient)
    with torch.no_grad():
        # Looping over training batches
        for X, Y, _ in testLoader:
            X, Y = X.float(), Y.float()

            yHat = model(X)
            loss = lossFunc(yHat, Y)

            totalTestLoss += loss.item()
    testLoss = totalTestLoss / len(testLoader)
    testLosses.append(testLoss)

    """ Plotting """
    # Update the plot data
    trainLine.set_data(range(len(trainLosses)), trainLosses)
    testLine.set_data(range(len(testLosses)), testLosses)

    # Rescale axes
    ax.relim()
    ax.autoscale_view()

    # Redraw the plot
    plt.draw()
    plt.pause(0.01)  # tiny pause to allow refresh

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {trainLoss:.6f} | Test Loss: {testLoss:.6f}")

    if epoch % 50 == 0:  # Every 50 epochs, we plot the signal vs the predicted signal
        X, Y, _ = dataset.getTrial()
        yHat = model(X)

        # Move to CPU for plotting
        Y = Y.cpu().detach().numpy()
        yHat = yHat.cpu().detach().numpy()

        figs, axs = plt.subplots(3, 1)
        fig.suptitle(f"Predicted vs True Signal at Epoch {epoch}")

        axs[0].plot(yHat[:, 0], label='Predicted Signal')
        axs[0].plot(Y[:, 0], label='Signal')
        axs[0].set_ylabel('Anteroposterior Displacement')
        axs[0].set_xlabel('Time')
        axs[0].legend()


        axs[1].plot(yHat[:, 1], label='Predicted Signal')
        axs[1].plot(Y[:, 1], label='Signal')
        axs[1].set_ylabel('Mediolateral Displacement')
        axs[1].set_xlabel('Time')
        axs[1].legend()

        axs[2].plot(yHat[:, 2], label='Predicted Signal')
        axs[2].plot(Y[:, 2], label='Signal')
        axs[2].set_ylabel('Vertical Displacement')
        axs[2].set_xlabel('Time')
        axs[2].legend()

        plt.show()




plt.ioff()  # Turn off interactive mode
plt.show()  # Final display if you want a clean static plot at the end