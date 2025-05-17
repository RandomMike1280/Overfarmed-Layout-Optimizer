import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from summary import ModelSummary

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        # input -> 3x3 convolutional filters -> batch norm -> relu
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        # input -> 3x3 convolutional filters -> batch norm -> ReLU -> 3x3 convolutional filters -> batch norm  -> Skip connection -> ReLU
        super(ResNet, self).__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = ConvLayer(out_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        residual = x
        x = self.relu(self.bn(self.conv1(x)))
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x + residual)
        return x

class ValueHead(nn.Module):
    def __init__(self, in_channels, state_size, tanh):
        # input -> 1 1x1 convolutional filters -> batch norm -> ReLU -> Fully connected -> ReLU -> Fully connected -> optional(tanh)
        super(ValueHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1, 1, 0)
        self.bn = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(state_size[0] * state_size[1], 256)
        self.fc2 = nn.Linear(256, 1)
        self.tanh = nn.Tanh() if tanh else nn.Identity()
        
    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.tanh(self.fc2(x))
        return x

class PolicyHead(nn.Module):
    def __init__(self, in_channels, state_size, action_size):
        # input -> 2 1x1 convolutional filters -> batch norm -> ReLU -> Fully connected
        super(PolicyHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, 2, 1, 1, 0)
        self.bn = nn.BatchNorm2d(2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(state_size[0] * state_size[1] * 2, action_size)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class AlphaFarmer(nn.Module):
    def __init__(self, in_channels, state_size, action_size):
        super(AlphaFarmer, self).__init__()
        self.conv = ConvLayer(in_channels, 64, 3, 1, 1)
        self.resnet = nn.ModuleList([ResNet(64, 64, 3, 1, 1) for _ in range(5)])
        self.value_head = ValueHead(64, state_size, False)
        self.policy_head = PolicyHead(64, state_size, action_size)

    def forward(self, x):
        x = self.conv(x)
        for resnet in self.resnet:
            x = resnet(x)
        value = self.value_head(x)
        policy = self.policy_head(x)
        return value, policy

if __name__ == '__main__':
    model = AlphaFarmer(5, (8, 8), 5)
    ModelSummary(model, input_size=(5, 8, 8))
    dummy_input = torch.randn(1, 5, 8, 8)
    output = model(dummy_input)
    print(output)