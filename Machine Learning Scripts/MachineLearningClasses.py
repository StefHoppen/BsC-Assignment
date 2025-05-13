import os
import torch
import math



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

""" BASE MACHINE LEARNING CLASSES """
class FNN(torch.nn.Module):
    """ Base class of a FeedForward Neural Network (FNN) """
    shape_options = ['Flat', 'ExCo', 'Contract', 'Expand']
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
                    3. 'Contract: Will contract the hidden size gradually to the output size.
                    4. 'Expand: Opposite of contract '

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

        if (shape == 'Flat' or shape == 'Contract') and input_size < output_size:
            raise ValueError(f'Flat/Contract shape is only supported when output size is smaller than the input size')

        if shape == 'Expand' and  output_size < input_size:
            raise ValueError(f'Expand shape is only supported when output size is bigger than the input size')

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
        else: # Shape = Contract or Expand
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


""" AutoEncoders """
class FNNAutoEncoder(torch.nn.Module):
    """ Auto Encoder based on Feed Forward Neural Networks"""
    imu_dim = 6  # Number of features each IMU produces per time step
    save_dir = r'C:\Users\stefh\Documents\ME Year 3\BSC Assignment\GitHub Repository\Machine Learning Scripts\AutoEncoderSaveStates'

    def __init__(self, n_layers: int, window_size: int, compression_ratio:float, act_func:str):
        """ INPUTS:
                1. n_layers: Number of hidden layers in the ENCODER
                2. window_size: Number of time steps of IMU data that is used in the input
                3. compression ratio: input_size / latent_size """
        super(FNNAutoEncoder, self).__init__()
        self.input_size = self.imu_dim*window_size
        self.latent_size = round(self.input_size / compression_ratio)

        self.encoder = FNN(input_size=self.input_size, output_size=self.latent_size,
                           n_hidden=n_layers, shape='Contract', act_func=act_func)
        self.decoder = FNN(input_size=self.latent_size, output_size=self.input_size,
                           n_hidden=n_layers, shape='Expand', act_func=act_func)

        self.filename = f'FNNAutoEnc_l{n_layers}_w{window_size}_c{compression_ratio}_{act_func}.pth'
        self.path = os.path.join(self.save_dir, self.filename)

    def forward(self, x:torch.tensor):
        latent = self.encoder(x)  # Compute latent representation
        recon = self.decoder(latent)  # Extract reconstruction from latent representation
        return latent, recon

    def saveParams(self):
        torch.save(self.state_dict(), self.path)
        print(f"Model parameters saved to {self.path}")

    def loadParams(self):
        if os.path.exists(self.path):
            self.load_state_dict(torch.load(self.path))
            print(f"Model parameters loaded from {self.path}")
        else:
            print(f"No saved model found at {self.path}")


class FeedForwardNeuralNetwork(torch.nn.Module):
    activation_options = ['ReLU', 'Tanh', 'Sigmoid']
    def __init__(self, architecture: list, act_func: str, p_dropout: float):
        super(FeedForwardNeuralNetwork, self).__init__()
        self.architecture = architecture

        layers = []
        for i in range(1, len(architecture)):
            in_ = self.architecture[i-1]
            out_ = self.architecture[i]
            layers.append(torch.nn.Linear(in_features=in_, out_features=out_))
            if act_func == 'ReLU':
                layers.append(torch.nn.ReLU())
            elif act_func == 'Tanh':
                layers.append(torch.nn.Tanh())
            elif act_func == 'Sigmoid':
                layers.append(torch.nn.Sigmoid())
            else:
                raise ValueError('A wrong activation function is passed')
            layers.append(torch.nn.Dropout(p=p_dropout))
        del layers[-2:]  # Final layers have no dropout or activation function

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x: torch.tensor):
        """ Forward method of the model """
        return self.model(x)

class CausalConv1D(torch.nn.Module):
    """ Custom convolution class that does not 'peek' in the future. This also includes weight normalisation """
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(CausalConv1D, self).__init__()
        self.padding = (kernel_size-1)*dilation
        self.conv = torch.nn.utils.parametrizations.weight_norm(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size,
                      dilation=dilation, padding=0))

    def forward(self, x):
        # Apply the causal padding to the past side of the signal
        x = torch.nn.functional.pad(x, (self.padding, 0), mode='constant', value=0)
        # Apply convolution
        out = self.conv(x)
        return out

class ResidualBlock(torch.nn.Module):
    """ Creates a Temporal Convolution Block.
        The block consists of:
        2x
            1. Dilated Causal Convolution
            2. Weight Normalisation
            3. ReLU activation
            4. Dropout Layer
        9. Residual connection
        10. ReLU Activation """

    def __init__(self, in_channels, channels, kernel_size, dilation, p_dropout):
        super(ResidualBlock, self).__init__()
        self.Conv1 = CausalConv1D(in_channels, channels, kernel_size, dilation)
        self.Conv2 = CausalConv1D(channels, channels, kernel_size, dilation)
        self.ReLU = torch.nn.ReLU()
        self.Dropout = torch.nn.Dropout(p_dropout)

        # Optional 1x1 conv if input and output channels differ
        self.Downsample = torch.nn.Conv1d(in_channels, channels, kernel_size=1) \
            if in_channels != channels else None

    def forward(self, x: torch.tensor):
        out = self.Conv1(x)
        out = self.ReLU(out)
        out = self.Dropout(out)

        out = self.Conv2(out)
        out = self.ReLU(out)
        out = self.Dropout(out)

        # Residual connection
        res = x if self.Downsample is None else self.Downsample(x)
        return self.ReLU(out + res)

class TemporalConvolutionNetwork(torch.nn.Module):
    """ Creates a Temporal Convolution Network """
    def __init__(self, in_channels, dilations: list, channels:list, kernel_size:int, p_dropout, outputs: int):
        super(TemporalConvolutionNetwork, self).__init__()
        self.in_channels = in_channels
        self.dilations = dilations
        self.channels = channels

        self.receptive_field = 1
        for val in self.dilations:
            self.receptive_field += 2*(kernel_size-1) * val
        self.receptive_field -= (kernel_size-1) * self.dilations[-1]  # Final layer does not add receptive field

        layerBlocks = []
        in_channels = self.in_channels
        for idxLayer in range(len(self.dilations)):
            dilation = self.dilations[idxLayer]
            out_channels = self.channels[idxLayer]
            residualBlock = ResidualBlock(in_channels=in_channels, channels=out_channels,
                                          kernel_size=kernel_size, dilation=dilation, p_dropout=p_dropout)
            layerBlocks.append(residualBlock)
            in_channels = out_channels

        self.resBlocks = torch.nn.Sequential(*layerBlocks)
        self.Linear = torch.nn.Linear(in_features=self.channels[-1], out_features=outputs)


    def forward(self, x: torch.tensor):
        x = self.resBlocks(x) # Size = [Batch x Channels x Times]
        x = x.permute(0, 2, 1) # Size = [Batch x Times x Channels]
        x = self.Linear(x)  # Size = [Batch x Times x Output]
        x = x.permute(0, 2, 1) # Size = [Batch x Output x Times]
        return x

