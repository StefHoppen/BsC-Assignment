import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataDirectory = r'C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Data Files\StraightWalking\Python\MachineLearningData.json'

with open(dataDirectory, 'r') as json_file:
    allData = json.load(json_file)


""" IMU Measurements """
# Creating a PD Dataframe for IMU measurements
allFrameStorage = []
for patientID, patientData in allData.items():
    for trialID, trialData in patientData.items():
        trialFrameStorage = []
        for sensorID, sensorData in trialData['IMU'].items():
            columns = [
                f'{sensorID} accX', f'{sensorID} accY', f'{sensorID} accZ',
                f'{sensorID} gyrX', f'{sensorID} gyrY', f'{sensorID} gyrZ'
            ]
            sensorList = []
            for i in range(len(sensorData['Accelerometer'])):
                accX, accY, accZ = sensorData['Accelerometer'][i]
                gyrX, gyrY, gyrZ = sensorData['Gyroscope'][i]
                dataList = [accX, accY, accZ, gyrX, gyrY, gyrZ]
                sensorList.append(dataList)

            sensorFrame = pd.DataFrame(sensorList, columns=columns)
            trialFrameStorage.append(sensorFrame)
        trialFrame = pd.concat(trialFrameStorage, axis=1)

        # Adding Time
        time =  [i * 0.01 for i in range(len(trialFrame))]
        trialFrame['Time'] = time

        # Adding Patient and Trial info
        trialFrame['Patient'] = patientID
        trialFrame['Trial'] = trialID

        # Storing Trial in a list
        allFrameStorage.append(trialFrame)
TotalFrame = pd.concat(allFrameStorage, axis=0)

# Z Score normalisation
# 1. Identify columns that belong to sensor data
sensorColumns = TotalFrame.columns.difference(['Time', 'Patient', 'Trial'])

# 2. Group by axis type (accX, accY, etc.)
measurementTypes = ['accX', 'accY', 'accZ', 'gyrX', 'gyrY', 'gyrZ']

for type_ in measurementTypes:
    # Get all columns that end with this axis type
    matchingCols = [col for col in sensorColumns if col.endswith(type_)]

    # Flatten all values from these columns into one array
    allValues = pd.concat([TotalFrame[col] for col in matchingCols], axis=0)
    mean = allValues.mean()
    std = allValues.std()

    # Apply z-score normalization to each matching column
    for col in matchingCols:
        TotalFrame[col] = (TotalFrame[col] - mean) / std

# Saving and clearing the frame
TotalFrame.to_csv(r'C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Data Files\StraightWalking\Python\IMU.csv', index=False)
TotalFrame = pd.DataFrame()

""" CoM Position """
columns = ['Anteroposterior', 'Mediolateral', 'Vertical']
trialFrameStorage = []
for patientID, patientData in allData.items():
    for trialID, trialData in patientData.items():
        trialFrame = pd.DataFrame(trialData['COM'], columns=columns)
        trialFrame = trialFrame-trialFrame.mean()

        trialFrame['Patient'] = patientID
        trialFrame['Trial'] = trialID
        trialFrameStorage.append(trialFrame)

TotalFrame = pd.concat(trialFrameStorage, axis=0)

# Saving and clearing the frame
TotalFrame.to_csv(r'C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Data Files\StraightWalking\Python\COM.csv', index=False)
TotalFrame = pd.DataFrame()


""" Ground Reaction Forces """
columns = ['Anteroposterior', 'Mediolateral', 'Vertical']
trialFrameStorage = []
for patientID, patientData in allData.items():
    for trialID, trialData in patientData.items():
        trialFrame = pd.DataFrame(trialData['GRF'], columns=columns)
        trialFrame['Patient'] = patientID
        trialFrame['Trial'] = trialID
        trialFrameStorage.append(trialFrame)

TotalFrame = pd.concat(trialFrameStorage, axis=0)
# Saving the frame
TotalFrame.to_csv(r'C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Data Files\StraightWalking\Python\GRF.csv', index=False)