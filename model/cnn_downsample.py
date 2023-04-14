import torch
import torch.nn as nn

class CNN_downsample(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            features,
            stride,
            kernel_size,
    ):
        """
        Parameters:
        -----------
        in_channels: number of input channels (sequence length in this application)
        features: list of channels to downsample to with multiple Conv1d layers
        stride: stride of Conv1d
        kernel_size: kernel size for Conv1d (min 3)
        """
        super(CNN_downsample, self).__init__()
        self.CNN_downsample = nn.ModuleList()

        for feature in features:
            self.CNN_downsample.append(
                nn.Conv1d(
                in_channels=in_channels,
                out_channels=feature,
                kernel_size=kernel_size,
                stride=stride
            ))
            self.CNN_downsample.append(
                nn.ReLU()
            )

    def forward(self, input):
        """
        Parameters:
        -----------
        input: input tensor (B, L, C)

        Returns:
        output:  output downsampled tensor (B, L, C)
        --------
        """
        torch.permute(input, (0, 2, 1))
        for layer in self.CNN_downsample:
            input = layer(input)
        output = input
        torch.permute(output, (0, 2, 1))
        return output