import torch
import torch.nn as nn
import torch.nn.functional as F

class block(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            stride: int = 1
        ):
        """
        Parameters:
        -----------
        in_channels: number of input channels
        out_channels: number of output channels
        stride: stride for convolution
        """
        super(block, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Skip connection (identity mapping) if input and output channels differ
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
            
    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output += self.skip(input) # skip connection
        output = self.relu(output)
        return output

# 1D ResNet18 model
class ResNet18_1D(nn.Module):
    def __init__(
            self, 
            num_classes: int
        ):
        """
        Parameters:
        -----------
        num_classes: number of features in a seq2seq task
        """
        super(ResNet18_1D, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(9, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(out_channels=64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        """
        Parameters:
        -----------
        out_channels: number of out channels in layer
        num_blocks: number of blocks within layer
        stride: stride for blocks within layer

        Returns:
        --------
        nn.Sequential of the sum of layers of the ResNet network 
        """
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1) # reshape tensor
        x = self.fc(x)
        return x