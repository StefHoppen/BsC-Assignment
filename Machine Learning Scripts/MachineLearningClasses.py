import torch

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

