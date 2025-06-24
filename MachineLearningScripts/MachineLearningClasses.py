import numpy as np
import torch
import math

def polyFit(y: torch.Tensor, poly_deg: int, f_sample: int):
    """
    Fits an N-th degree polynomial to each timeseries axis (x, y, z), while respecting gradient flow.

    # y_1 = a_0 + a_1*x_1 + a_2*x_1^2
    # Y = [X] A
    #   Y = Column vector of data you want to fit to
    #   X = Matrix with x coefficients ([1 x x^2 ... x^poly_deg]
    #   A = Column vectors of a coefficients ([a_0 a_1 a_2 ... a_poly_deg])

        Inputs:
            y: Tensor of shape (Batch x Axis x Time steps)
            poly_deg: Degree of the polynomial that is fitted
            f_sample: Sampling frequency in Hz of the input signal.
        Returns:
            y_fitted: Tensor same shape as

    """
    nBatch, nAxis, nPoints = y.shape
    timeVec = torch.arange(nPoints) * (1/f_sample)

    # Build the Vandermonde matrix
    X = torch.stack([timeVec ** i for i in range(poly_deg + 1)], dim=1)

    allCoefficients = torch.zeros((nBatch, nAxis, poly_deg+1))
    yFitted = torch.zeros((nBatch, nAxis, nPoints))
    for iBatch in range(nBatch): # Loop over each batch
        for iAxis in range(nAxis):  # Loop over x, y, z
            Y = y[iBatch, iAxis, :]
            A, *_ = torch.linalg.lstsq(X, Y)  # This is weird notation, but check it.
            allCoefficients[iBatch, iAxis, :] = A
            yFitted[iBatch, iAxis, :] = torch.matmul(X, A)
    return yFitted


""" UNUSED CLASSES """
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

class GaussianConv1D(torch.nn.Module):
    """ Uses a Gaussian Kernel to smooth out a 1D signal. """
    def __init__(self, window_size: int, std_dev:float):
        super(GaussianConv1D, self).__init__()
        self.window_size = window_size
        self.std_dev = std_dev
        self.const_term = 1 / (std_dev*math.sqrt(2*math.pi))  # Constant term of the normal distribution
        kernel = self.constructKernel()
        self.register_buffer('kernel', kernel.view(1, 1, -1))  # shape: (out_channels, in_channels, kernel_size)
        #self.kernel = torch.nn.Parameter(kernel.view(1, 1, -1), requires_grad=True)  # Learnable!

    def forward(self, x:torch.tensor):
        # Looping over axis
        allAcc = []
        for axis in range(x.shape[1]):
            axisPos = x[:, axis:axis+1, :]
            axisAcc = torch.nn.functional.conv1d(axisPos, self.kernel)
            allAcc.append(axisAcc)
        allAcc = torch.cat(allAcc, dim=1)
        return allAcc

    def sampleDist(self, x: float):
        power = - x**2 / (2*self.std_dev**2)
        y = self.const_term * math.exp(power)
        return y

    def constructKernel(self):
        R = self.window_size // 2
        xRange = list(range(-R, R+1))
        if self.window_size % 2 == 0:  # If window size is even
            raise ValueError('Window size must be an uneven integer')

        kernelCoefs = []
        for x in xRange:
            yNormal = self.sampleDist(x)
            kernelCoefs.append(yNormal)
        kernelCoefs = torch.tensor(kernelCoefs)  # Create tensor
        kernelCoefs /= torch.sum(kernelCoefs)  # Normalise
        return kernelCoefs


""" PINN EXTENSION """
class RateLimiter(torch.nn.Module):
    """ Reduces the sampling rate of an output signal through taking an average """
    def __init__(self, n_points:int):
        super(RateLimiter, self).__init__()
        self.n_points = n_points
        coefficient = torch.tensor(1/n_points)
        kernel = torch.stack([coefficient]*n_points)
        self.register_buffer('kernel', kernel.view(1, 1, -1))

    def forward(self, x):
        # Looping over axis
        newSignalList = []
        for axis in range(x.shape[1]):
            oldSignal = x[:, axis:axis + 1, :]
            newSignal = torch.nn.functional.conv1d(oldSignal, self.kernel, stride=self.n_points)
            newSignalList.append(newSignal)
        newSignalTensor = torch.cat(newSignalList, dim=1)
        return newSignalTensor

class SimpleFDConv1D(torch.nn.Module):
    """ Simple Central Finite Difference calculator that computes the 2nd order derivative """
    kernel_coefficients = {
        1: torch.tensor([1, -2, 1], dtype=torch.float32),
        2: torch.tensor([-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12], dtype=torch.float32),
        3: torch.tensor([1 / 90, -3 / 20, 3 / 2, -49 / 18, 3 / 2, -3 / 20, 1 / 90], dtype=torch.float32),
        4: torch.tensor([-1 / 560, 8 / 315, -1 / 5, 8 / 5, -205 / 72, 8 / 5, -1 / 5, 8 / 315, -1 / 560], dtype=torch.float32)
    }

    def __init__(self, stencil_radius: int, h:float):
        super(SimpleFDConv1D, self).__init__()
        self.R = stencil_radius
        self.h = h
        kernel = self.kernel_coefficients[stencil_radius] / (h**2)
        # The following line tells PyTorch to not train these parameters, but do save them when saving the model
        self.register_buffer('kernel', kernel.view(1, 1, -1))  # shape: (out_channels, in_channels, kernel_size)
        #self.kernel = torch.nn.Parameter(kernel.view(1, 1, -1), requires_grad=True)  # Now learnable!


    def forward(self, x:torch.tensor):
        # Looping over axis
        allAcc = []
        for axis in range(x.shape[1]):
            axisPos = x[:, axis:axis+1, :]
            axisAcc = torch.nn.functional.conv1d(axisPos, self.kernel)
            allAcc.append(axisAcc)
        allAcc = torch.cat(allAcc, dim=1)
        return allAcc

class PINNExtension(torch.nn.Module):
    """ This class combines the SimpleFDConv1D and GaussianConv1D """
    def __init__(self,
                 fd_stencil_radius: int, fd_h: float,
                 gauss_window_size: int, gauss_std: float):
        super(PINNExtension, self).__init__()
        self.differentiator = SimpleFDConv1D(stencil_radius=fd_stencil_radius,
                                             h=fd_h)
        self.smoother = GaussianConv1D(window_size=gauss_window_size,
                                       std_dev=gauss_std)

    def forward(self, x: torch.tensor):
        xDiff = self.differentiator(x)
        xSmooth = self.smoother(xDiff)
        return xSmooth

"""" STANDARD TCN """
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
