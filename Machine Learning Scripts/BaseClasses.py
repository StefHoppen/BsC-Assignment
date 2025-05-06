import pandas as pd
import os
import torch
from torch.utils.data import Dataset
import math
import json
import random


def computeLinearArchitecture(input_size, steps, output_size):
    b = input_size
    a = (output_size - b) / steps

    architecture = []
    previous_size = input_size
    for idx in range(1, steps):
        size = round(a*idx + b)
        architecture.append((previous_size, size))
        previous_size = size
    # Adding final layer
    architecture.append((previous_size, output_size))
    return architecture

class FNN(torch.nn.Module):
    """ Base class of a FeedForward Neural Network (FNN) """
    shape_options = ['Flat', 'ExCo', 'Bottleneck']
    activation_options = ['ReLU', 'Tanh', 'Sigmoid']
    def __init__(self, input_size: int, n_hidden: int, output_size: int, shape:str, act_func: str):
        """ INPUTS:
            1. input_size = Number of features in the input (int)
            2. n_hidden = Number of hidden layers in the network (int)
            3. output_size = Number of features in the output (int)
            4. shape = Determines how big the hidden layers are (str).
               Options:
                    1. 'Flat': Will keep the hidden layer size constant.
                    2. 'ExCo': Expand -> Contract: In the first half of the network, the number of neurons will expand
                        gradually to twice the size of the input. In the second half, it will contract gradually to the
                        output size.
                    3. 'Bottleneck: Will contract the hidden size gradually to the output size.'
            5. act_func = Activation function
                Options:
                    1. ReLU - Rectified Linear Unit
                    2. Tanh - Hyperbolic tangent, ranges from -1 to 1
                    3. Sigmoid - Ranges from 0 to 1 """

        super(FNN, self).__init__()
        """ Preparation for building network """
        # Checking if shape is supported
        if not shape in self.shape_options:
            raise ValueError(f'\'{shape}\' is not a supported network shape. Options: {self.shape_options}')

        if (shape == 'Flat' or shape == 'Bottleneck') and input_size < output_size:
            raise ValueError(f'Flat/Bottleneck shape is only supported when output size is smaller than the input size')

        # Checking if activation is supported
        if not act_func in self.activation_options:
            raise ValueError(f'\'{act_func}\' is not a supported activation function. Options: {self.activation_options}')

        self.input_size = input_size
        self.n_hidden = n_hidden
        self.output_size = output_size
        self.shape = shape
        self.act_func = act_func

        # Constructing architecture
        self.architecture = []
        if shape == 'Flat':
            self.architecture = [(input_size, input_size),] * n_hidden # Hidden layers
            self.architecture.append((input_size, output_size))  # Final layers

        elif shape == 'ExCo':
            exp_size = input_size * 2
            middle_idx = math.ceil(n_hidden/2)
            exp_arch = computeLinearArchitecture(input_size=input_size, steps=middle_idx, output_size=exp_size)
            cont_arch = computeLinearArchitecture(input_size=exp_size, steps=middle_idx, output_size=output_size)

            if n_hidden % 2 == 0:  # If number of layers is even (Means that in the middle we have a straight section)
                extra_layer = [(exp_size, exp_size)]
                self.architecture = exp_arch + extra_layer + cont_arch
            else:
                self.architecture = exp_arch + cont_arch
        else: # Shape = Bottleneck
            self.architecture = computeLinearArchitecture(input_size=input_size, steps=n_hidden, output_size=output_size)

        """ Constructing the network """
        layers = []
        for in_, out_ in self.architecture:
            layers.append(torch.nn.Linear(in_features=in_, out_features=out_))
            if act_func == 'ReLU':
                 layers.append(torch.nn.ReLU())
            elif act_func == 'Tanh':
                layers.append(torch.nn.Tanh())
            elif act_func == 'Sigmoid':
                layers.append(torch.nn.Sigmoid())
            else:
                raise ValueError('This should not happen: A wrong activation is passed and not detected')
        layers.pop() # Final layer nas no activation

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x: torch.tensor):
        """ Forward method of the model """
        return self.model(x)

class StraightWalkingDataset(Dataset):
    def __init__(self, dev):
        """ Initialises the dataset """
        self.directory = r'C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Data Files\StraightWalking\Python'

        # Load data
        imuDF = pd.read_csv(os.path.join(self.directory, 'IMU.csv'))
        comDF = pd.read_csv(os.path.join(self.directory, 'COM.csv'))
        grfDF = pd.read_csv(os.path.join(self.directory, 'GRF.csv'))

        # Column selection
        self.imuCols = [col for col in imuDF.columns if col not in ['Time', 'Patient', 'Trial']]
        self.comCols = [col for col in comDF.columns if col not in ['Time', 'Patient', 'Trial']]
        self.grfCols = [col for col in grfDF.columns if col not in ['Time', 'Patient', 'Trial']]

        # Convert to tensors once
        self.IMU = torch.tensor(imuDF[self.imuCols].to_numpy(), dtype=torch.float32).to(dev)
        self.COM = torch.tensor(comDF[self.comCols].to_numpy(), dtype=torch.float32).to(dev)
        self.GRF = torch.tensor(grfDF[self.grfCols].to_numpy(), dtype=torch.float32).to(dev)

        # Extract indices related to trials
        self.trialIdxMap = {}
        for idx, row in imuDF.iterrows():
            key = (row['Patient'], row['Trial'])
            if key not in self.trialIdxMap:
                self.trialIdxMap[key] = []
            self.trialIdxMap[key].append(idx)


    def __len__(self):
        return self.IMU.shape[0]

    def __getitem__(self, idx):
        return self.IMU[idx], self.COM[idx], self.GRF[idx]

    def getTrial(self):
        """ Returns data of random trial"""
        randomKey = random.choice(list(self.trialIdxMap.keys()))
        trialIndices = self.trialIdxMap[randomKey]
        return self.IMU[trialIndices, :], self.COM[trialIndices, :], self.COM[trialIndices, :]