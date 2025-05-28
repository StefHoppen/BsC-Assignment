import json
import pandas as pd
import os
import pickle

# Specifying file path
dataPath = r'C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Data Files\StraightWalking\Python\MachineLearningData.json'

# Reading data from file path
with open(dataPath, 'r') as json_file:
    allData = json.load(json_file)

imuStorage = {
        # Each sensor is stored in a separate field, this makes it more manageable later
        'LF': [],  # Left Foot
        'RF': [],  # Right Foot
        'Ste': [],  # Sternum
        'Pel': []  # Pelvis
        }
comStorage = []
grfStorage = []

""" === LOOP OVER PATIENTS """
for patientID, patientData in allData.items():
    """ === LOOP OVER TRIALS """
    for trialID, trialData in patientData.items():
        """ === LOOP OVER STEPS """
        for stepID, stepData in trialData.items():
            """ -- PROCESSING COM SIGNAL -- """
            comDataFrame = pd.DataFrame(stepData['COM'], columns=['x', 'y', 'z'])
            comDataFrame['Patient'] = patientID
            comDataFrame['Trial'] = trialID
            comDataFrame['Step'] = stepID
            comStorage.append(comDataFrame)

            """ -- PROCESSING GRF SIGNAL -- """
            grfDataFrame = pd.DataFrame(stepData['GRF'], columns=['x', 'y', 'z'])
            grfDataFrame['Patient'] = patientID
            grfDataFrame['Trial'] = trialID
            grfDataFrame['Step'] = stepID
            grfStorage.append(grfDataFrame)

            """ -- PROCESSING IMU SIGNAL -- """
            for sensorID, sensorData in stepData['IMU'].items():
                accDataFrame = pd.DataFrame(sensorData['Accelerometer'], columns=['acc_x', 'acc_y', 'acc_z'])
                gyrDataFrame = pd.DataFrame(sensorData['Gyroscope'], columns=['gyr_x', 'gyr_y', 'gyr_z'])
                sensorDataFrame = pd.concat([accDataFrame, gyrDataFrame], axis=1)
                imuStorage[sensorID].append(sensorDataFrame)

"""" === COMBINING DATAFRAMES === """
comStorage = pd.concat(comStorage, axis=0)
grfStorage = pd.concat(grfStorage, axis=0)
for sensorID, storageList in imuStorage.items():
    imuStorage[sensorID] = pd.concat(storageList, axis=0)

""" === NORMALISING IMU FEATURES === """
for sensorID, sensorDataFrame in imuStorage.items():  # Z-score normalise per sensor, per axis
    sensorNormalised = sensorDataFrame.copy()
    for col in sensorDataFrame.columns:
        mean = sensorDataFrame[col].mean()
        std = sensorDataFrame[col].std()
        sensorNormalised[col] = (sensorDataFrame[col] - mean) / std
    imuStorage[sensorID] = sensorNormalised

# Normalisation of the COM features happen in the Dataset Class, such that we can store the mean and std and transform signals back to meters.
""" === STORING FILES === """
comStorage.to_csv(r'C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Data Files\StraightWalking\Python\COM.csv', index=False)
grfStorage.to_csv(r'C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Data Files\StraightWalking\Python\GRF.csv', index=False)
for sensorID, sensorDataFrame in imuStorage.items():
    sensorDataFrame.to_csv(rf'C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Data Files\StraightWalking\Python\IMU_{sensorID}.csv', index=False)





