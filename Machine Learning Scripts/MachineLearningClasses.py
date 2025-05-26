import torch
import math

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

class CentralFDConv1D(torch.nn.Module):
    """ Uses a predefined filter to compute the second derivative of a signal, using Central Finite Difference method. """
    kernel_coefficients = {
        2: torch.tensor([1, -2, 1]),
        4: torch.tensor([-1/12, 4/3, -5/2, 4/3, -1/12]),
        6: torch.tensor([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90]),
        8: torch.tensor([-1/560, 8/315, -1/5, 8/5, -205/72, 8/5,  -1/5, 8/315, -1/560])
    }

    def __init__(self, stencil_radius: int, h: float = 0.01):
        """ Inputs:
            1. stencil_radius: How many datapoints on one side the Central Finite Difference method uses (must be greater than 2).
            2. h: Distance between two datapoints, in our case time.  """

        super(CentralFDConv1D, self).__init__()
        self.stencil_radius = stencil_radius
        self.kernel_size = stencil_radius*2 + 1
        self.kernel_weights = (self.kernel_coefficients[int(stencil_radius*2)].clone().view(1, 1, -1) / h**2).repeat(3, 1, 1)

        self.Conv = torch.nn.Conv1d(in_channels=3, out_channels=3, kernel_size=self.kernel_size, bias=False, groups=3)
        self.Conv.weight = torch.nn.Parameter(self.kernel_weights, requires_grad=False)

        # Calculating the weights for padding
        forwardMatrix, backwardMatrix = [], []
        for k in range(1, stencil_radius+1):
            forwardEntry, backwardEntry = [], []
            for m in range(1, stencil_radius+1):
                forwardEntry.append( math.pow(k, m) / math.factorial(m) )
                backwardEntry.append( math.pow(-k, m) / math.factorial(m) )
            forwardMatrix.append(forwardEntry)
            backwardMatrix.append(backwardEntry)

        self.forwardMatrix = torch.inverse(torch.tensor(forwardMatrix))  # For this matrix, we only need the inverse for padding
        self.backwardMatrix = torch.tensor(backwardMatrix)

    def padInputs(self, x: torch.tensor):
        """ Pads the input on both ends using Taylor Expansion """
        R = self.stencil_radius

        # Loop over each dimension separately
        startPadding = []
        endPadding = []
        for dimension in range(x.shape[1]):  # x has shape (Batch x Channels(dims) x Time steps)
            xStart = x[:, dimension, 0]
            xRight = x[:, dimension, 1:R+1].T # Make column vector
            xDiffStart = (xRight - xStart) # Make column vector
            padStartDim = xStart + self.backwardMatrix @ (self.forwardMatrix @ xDiffStart)
            padStartDim = torch.flip(padStartDim.T, (1,))
            startPadding.append(padStartDim)

            xEnd = x[:, dimension, -1]
            xLeft = x[:, dimension, -R-1:-1].T # Make column vector
            xLeft = torch.flip(xLeft, (0,))
            xDiffEnd = (xLeft - xEnd)
            padEndDim = xEnd + self.backwardMatrix @ (self.forwardMatrix @ xDiffEnd)
            padEndDim = padEndDim.T
            endPadding.append(padEndDim)

        startPadding = torch.stack(startPadding, dim=1)
        endPadding = torch.stack(endPadding, dim=1)
        xPadded = torch.cat([startPadding, x, endPadding], dim=2)
        return xPadded

    def forward(self, x: torch.tensor):
        x = self.padInputs(x)  # Pad the inputs
        second_derivative = self.Conv(x)
        return second_derivative

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

def PINNLoss(comHat: torch.tensor, comPos: torch.tensor , comAcc: torch.tensor, grf: torch.tensor, wPhysics:float):
    naive_loss = torch.mean((comHat - comPos)**2)
    physics_loss = torch.mean(torch.abs(comAcc - grf))

    total_loss = naive_loss + wPhysics*physics_loss
    return total_loss

def NaiveLoss(comHat: torch.tensor, comPos: torch.tensor):
    return torch.mean((comHat - comPos)**2)


