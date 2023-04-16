import torch
import torch.nn as nn

class CNN_downsample(nn.Module):
    def __init__(
            self,
            in_channels,
            channels,
            stride,
            kernel_size,
            seq_len,
    ):
        """
        Parameters:
        -----------
        in_channels: number of input channels (sequence length in this application)
        channels: list of number of channels corresponding to number of Conv1d layers
        stride: stride of Conv1d
        kernel_size: kernel size for Conv1d (min 3)
        seq_len: sequence lenth of data tensor
        """
        super(CNN_downsample, self).__init__()
        self.CNN_downsample = nn.ModuleList()
        self.seq_len = seq_len

        if isinstance(channels, int):
            channels = [channels]

        for channel in channels:
            self.CNN_downsample.append(
                nn.Conv1d(
                in_channels=in_channels,
                out_channels=channel,
                kernel_size=kernel_size,
                stride=stride,
                padding=(stride*(self.seq_len//2-1)+1+(kernel_size-1)-self.seq_len)//2
            ))
            self.CNN_downsample.append(
                nn.ReLU()
            )
            in_channels = channel

    def forward(self, input):
        """
        Parameters:
        -----------
        input: input tensor (B, L, C)

        Returns:
        output:  output downsampled tensor (B, L, C)
        --------
        """
        input = torch.permute(input, (0, 2, 1))
        for layer in self.CNN_downsample:
            input = layer(input)
        output = torch.permute(input, (0, 2, 1))
        return output