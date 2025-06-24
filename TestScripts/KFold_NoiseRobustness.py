import pandas as pd

from MachineLearningScripts.DataSets import StraightWalkingTCNSet
from MachineLearningScripts.FinalModels import *

import torch
from torch.utils.data import DataLoader, random_split
from itertools import chain

TRAIN_MODELS = True

randomGenerator = torch.Generator().manual_seed(16)
maxEpochs = 500  # Maximum epochs the model will execute before re-initialising
convergeRequirement = 0.95  # Required performance fraction for the model to be 'converged'
pinnWeights = [0, 1e-4, 1e-3, 1e-2, 1e-1]  # PINN weights that are being tested
noiseLevels = [0.05, 0.10, 0.25]

minLosses = {  # Since generator is the same, these should be reachable
    0: [0.1236, 0.0877, 0.0705, 0.0603, 0.0562],
    1e-4: [0.1378, 0.0902, 0.0747, 0.0550, 0.0595],
    1e-3: [0.1373, 0.0846, 0.0750, 0.0557, 0.0563],
    1e-2: [0.1437, 0.1213, 0.0989, 0.0831, 0.0779],
    1e-1: [0.4581, 0.3502, 0.3781, 0.3362, 0.3810]}

totalDataset = StraightWalkingTCNSet()
nFolds = 5
Fraction = 1 / nFolds
dataLengths = [Fraction for i in range(nFolds)]
foldList = random_split(totalDataset, dataLengths, generator=randomGenerator)

resultSheetPerf = []
for j in range(nFolds):  # Looping over fold configurations (Essentially Runs)
    print(f'Starting testing for Fold: {j+1}')
    testSet = foldList[j]  # Selecting new test set
    testLoader = DataLoader(testSet, batch_size=1, shuffle=True, generator=randomGenerator)  # Loading test set

    trainSets = foldList[:j] + foldList[j + 1:]  # Selecting train set
    trainLoaders = [DataLoader(s, batch_size=1, shuffle=True, generator=randomGenerator) for s in trainSets]  # Loading train set

    for wPinn in pinnWeights: # Looping over configurations
        print(f'Started testing for PINN weight: {wPinn}')
        if TRAIN_MODELS:  # Models are not trained yet, thus we need to train it first
            thresholdLoss = minLosses[wPinn][j] / convergeRequirement
            modelConvergence = False

            while True:
                # Creating a new fresh model each time the algorithm tries
                if wPinn == 0:
                    model = TcnFinal()
                else:
                    model = PinnFinal(wPinn)

                for i in range(maxEpochs):
                    combinedTrainLoader = chain.from_iterable(trainLoaders)  # Combining all Train Dataloaders,
                    # this creates an interator that is destroyed each time its used.

                    # Training
                    _ = model.trainEpoch(combinedTrainLoader)

                    # Testing
                    lossDict, _ = model.testEpoch(testLoader)

                    if lossDict['Weighted'] <= thresholdLoss:
                        modelConvergence = True
                        break # Break the epoch loop

                    if i % 100 == 0:
                        print(f'Epoch: {i}')
                        print(f'Needed Loss Gain: {thresholdLoss - lossDict['Weighted'] }')

                if modelConvergence:
                    print('Stopping training, full performance is reached')
                    model.saveParams(j)
                    break # Break the while loop
                else:
                    print(f'Print did not reach loss threshold in {maxEpochs} epochs')
                    continue  # Start at the top of the while loop again (new model initialisation)
        else:
            # Creating a new fresh model
            if wPinn == 0:
                model = TcnFinal()
            else:
                model = PinnFinal(wPinn)
            model.loadParams(foldIdx=j)  # Loading parameters

        # Assessing Noise Robustness
        print(f'Assessing noise performance for PINN weight {wPinn}')
        for lvl in noiseLevels:
            xMAE, yMAE, zMAE = 0, 0, 0  # Initialising variables for Mean Absolute Errors
            for (patientID, trialID, stepID, IMU, COM, GRF), (comMean, comStd) in testLoader:
                gaussNoise = torch.randn_like(IMU) * lvl  # We can just add standard Gaussian noise, since the IMU signals
                # are preprocessed to follow a standard normal distribution.
                IMU = IMU + gaussNoise

                comHat = model.predict(IMU)
                comHatMeter = comHat * comStd + comMean

                comMeter = COM.squeeze() * comStd + comMean

                testMAE = torch.mean(torch.abs(comMeter - comHatMeter).squeeze(), dim=1)  # Computing Mean Absolute error
                # Storing Mean Absolute Errors
                xMAE += testMAE[0].item()
                yMAE += testMAE[1].item()
                zMAE += testMAE[2].item()
            xPerf = xMAE / len(testLoader)
            yPerf = yMAE / len(testLoader)
            zPerf = zMAE / len(testLoader)

            noisePerf = [j, wPinn, lvl, xPerf, yPerf, zPerf]
            resultSheetPerf.append(noisePerf)
    # Each fold we store performance
    resultSheet = pd.DataFrame(resultSheetPerf)
    resultSheet.to_csv(rf'C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Results Storage\ResultSheet\NoiseFold{j}',
                       index=False)



