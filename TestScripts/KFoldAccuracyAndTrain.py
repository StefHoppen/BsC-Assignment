import pandas as pd

from MachineLearningScripts.DataSets import StraightWalkingTCNSet
from MachineLearningScripts.FinalModels import *

import torch
from torch.utils.data import DataLoader, random_split
from itertools import chain
import time


randomGenerator = torch.Generator().manual_seed(16)  # For reproducibility
accuracyEpochs = 10  # Number of epochs the model is trained to test minimum loss
convergeRequirement = 0.95  # Required performance fraction for the model to be 'converged'
pinnWeights = [0, 1e-4, 1e-3, 1e-2, 1e-1]  # PINN weights that are being tested

# Initialising total dataset
totalDataset = StraightWalkingTCNSet()
nFolds = 5
Fraction = 1 / nFolds
dataLengths = [Fraction for i in range(nFolds)]
foldList = random_split(totalDataset, dataLengths, generator=randomGenerator)

# Metrics that will be stored for Training and Testing
trainColumns = ['Fold','WPinn', 'Weighted Loss', 'Naive Loss', 'Informed Loss']
testColumns = ['Fold','WPinn', 'Weighted Loss', 'Naive Loss', 'Informed Loss', 'xAE', 'yAe', 'zAE', 'Time']
resultSheetPerf = []
for j in range(nFolds):  # Looping over fold configurations (Essentially Runs)
    print(f'Starting testing for Fold: {j+1}')
    testSet = foldList[j]  # Selecting new test set
    testLoader = DataLoader(testSet, batch_size=1, shuffle=True, generator=randomGenerator)  # Loading test set

    trainSets = foldList[:j] + foldList[j+1:]  # Selecting train set
    trainLoaders = [DataLoader(s, batch_size=1, shuffle=True, generator=randomGenerator) for s in trainSets]  # Loading train set

    foldTrainPerf, foldTestPerf = [], []  # Creating variables that will store the Dataframes of all the training data.
    # If I ever want to plot something(like losses over epochs), I dont need to re-run the whole script.
    for wPinn in pinnWeights: # Looping over configurations
        print(f'Started testing for PINN weight: {wPinn}')
        # Creating a new fresh model
        if wPinn == 0:
            model = TcnFinal()
        else:
            model = PinnFinal(wPinn)

        modelTrainPerf, modelTestPerf = [], []  # Here train and test performances of a single model will be stored.
        # These will later be transferred to dataframes and stored in 'foldTrainPerf' and 'foldTestPerf'.

        for k in range(accuracyEpochs):  # Loop over epoch
            combinedTrainLoader = chain.from_iterable(trainLoaders)  # Combining all Train Dataloaders,
            # this creates an interator that is destroyed each time its used.

            # Training
            startTime = time.time()
            lossDict = model.trainEpoch(combinedTrainLoader)
            epochTrainPerf = [
                j, wPinn,
                lossDict['Weighted'], lossDict['Naive'], lossDict['Informed']]
            modelTrainPerf.append(epochTrainPerf)

            # Testing
            lossDict, perfDict = model.testEpoch(testLoader)
            epochTime = time.time() - startTime

            epochTestPerf = [
                j, wPinn,
                lossDict['Weighted'], lossDict['Naive'], lossDict['Informed'],
                perfDict['MAE']['X'], perfDict['MAE']['Y'], perfDict['MAE']['Z'], epochTime]
            modelTestPerf.append(epochTestPerf)

            if k % 100 == 0:
                print(f'Epoch: {k}')
                print(lossDict['Weighted'])

        # After all epochs we store the model performances in the fold performances
        modelTrainPerf = pd.DataFrame(modelTrainPerf, columns=trainColumns)
        foldTrainPerf.append(modelTrainPerf)

        modelTestPerf = pd.DataFrame(modelTestPerf, columns=testColumns)
        foldTestPerf.append(modelTestPerf)


        # Finding the epoch with maximum performance (Lowest weighted test loss)
        bestEpochIdx = modelTestPerf['Weighted Loss'].idxmin()
        bestEpoch = modelTestPerf.iloc[bestEpochIdx, 0:8] # Remove the EpochTime

        # Finding the number of epochs to get to 95% performance and average epoch time
        convergenceThreshold = bestEpoch['Weighted Loss'] / convergeRequirement
        convergenceIdx = modelTestPerf[modelTestPerf['Weighted Loss'] <= convergenceThreshold].index[0]
        averageEpochTime = modelTestPerf['Time'].mean()

        finalModelPerformance = [*bestEpoch.to_list(), convergenceIdx+1, averageEpochTime]
        resultSheetPerf.append(finalModelPerformance)

    # Storing the performances that will be stored in an Excel sheet, for graph making
    columns = ['Fold','WPinn', 'Weighted Loss', 'Naive Loss', 'Informed Loss', 'xAE', 'yAe', 'zAE',
               'Convergence Epochs', 'Epoch Time']
    #resultSheet = pd.DataFrame(resultSheetPerf, columns=columns)
    #resultSheet.to_csv(rf'C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Results Storage\ResultSheet\AccuracyAndTrainFold{j}.csv',
                       #index=False)

    # Storing intermediate values (for maybe later use)
    foldTrainPerf = pd.concat(foldTrainPerf, axis=0)
    #foldTrainPerf.to_csv(rf'C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Results Storage\IntermediateResults\TrainFold{j}.csv')

    foldTestPerf = pd.concat(foldTestPerf, axis=0)
    print('hoi')
    #foldTestPerf.to_csv(rf'C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Results Storage\IntermediateResults\TestFold{j}.csv')
print('Finished all testing!!!!!')
