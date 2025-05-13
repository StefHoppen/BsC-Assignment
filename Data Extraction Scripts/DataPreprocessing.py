import json
import pandas as pd
import os

IMU = True
COM = True
GRF = True

dataDirectory = r'C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Data Files\StraightWalking\Python\MachineLearningData.json'

with open(dataDirectory, 'r') as json_file:
    allData = json.load(json_file)

""" IMU Measurements """
if IMU:
    # Each sensor is stored in a separate DataFrame, this is easier for switching to AutoEncoders
    trialStorage = {
        'LF': [],
        'RF': [],
        'Ste': [],
        'Pel': [] }

    # Loop over Patients
    for patientID, patientData in allData.items():
        # Loop over Trials
        for trialID, trialData in patientData.items():
            # Loop over sensors
            for sensorID, sensorData in trialData['IMU'].items():
                tempStorage = [] # Storing all measurements in a trial for a sensor
                for t in range(len(sensorData['Accelerometer'])):
                    accX, accY, accZ = sensorData['Accelerometer'][t]
                    gyrX, gyrY, gyrZ = sensorData['Gyroscope'][t]
                    dataList = [accX, accY, accZ, gyrX, gyrY, gyrZ]
                    tempStorage.append(dataList)
                tempDF = pd.DataFrame(tempStorage) # Converting to a Pandas Dataframe
                tempDF['Time'] = [i * 0.01 for i in range(len(tempDF))]  # Adding time
                tempDF['Patient'] = patientID  # Adding patient
                tempDF['Trial'] = trialID
                trialStorage[sensorID].append(tempDF)  # Storing the trial data

    # Z-Normalisation:
    # We want to normalise across measurement type.
    # So for example: All acceleration in x-directions of all trials, by all sensors, is normalised by Z-score
    allMeasurements = pd.concat([pd.concat(trial) for trial in trialStorage.values()])  # Concatenate all measurements into
    # one large Pandas Dataframe
    normCols = allMeasurements.columns[:6]  # Get the columns of the measurements (PatientID etc. are excluded)
    allMeans = allMeasurements[normCols].mean()  # Getting the mean of each column
    allStd = allMeasurements[normCols].std()  # Getting the standard deviation of each column

    # Applying Z-Score normalisation to each trial separately
    for sensorID in trialStorage:  # Loop over sensors
        normalisedDataFrames = []
        for trial in trialStorage[sensorID]:  # Loop over each trial
            normalisedDF = trial.copy()  # Make a copy of the original DataFrame
            normalisedDF[normCols] = (trial[normCols] - allMeans) / allStd  # Normalise the copy
            normalisedDataFrames.append(normalisedDF)  # Store the normalised copy
        allTrialFrame = pd.concat(normalisedDataFrames)  # Concatenate all trials in 1 big frame
        # Saving the frame of each sensor to a seperate file
        allTrialFrame.to_csv(rf'C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Data Files\StraightWalking\Python\IMU_{sensorID}.csv', index=False)


""" CoM Position """
if COM:
    TotalFrame = pd.DataFrame()  # Clearing Frame
    columns = ['Anteroposterior', 'Mediolateral', 'Vertical']
    frameStorage = []
    for patientID, patientData in allData.items():
        for trialID, trialData in patientData.items():
            trialFrame = pd.DataFrame(trialData['COM'], columns=columns)
            trialFrame = trialFrame-trialFrame.mean()

            trialFrame['Patient'] = patientID
            trialFrame['Trial'] = trialID
            frameStorage.append(trialFrame)

    TotalFrame = pd.concat(frameStorage, axis=0)

    # Saving and clearing the frame
    TotalFrame.to_csv(r'C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Data Files\StraightWalking\Python\COM.csv', index=False)


""" Ground Reaction Forces """
if GRF:
    TotalFrame = pd.DataFrame()  # Clearing Frame
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


