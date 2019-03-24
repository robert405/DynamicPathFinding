import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualModuleDouble(nn.Module):

    def __init__(self, inputSize, outputSize):

        super(ResidualModuleDouble, self).__init__()

        self.conv1 = nn.Conv2d(inputSize, inputSize, 3, padding=1)
        self.conv2 = nn.Conv2d(inputSize, outputSize, 3, padding=1)

    def forward(self, inputData):

        residual = torch.cat((inputData, inputData), 1)
        h = F.elu(self.conv1(inputData))
        h = self.conv2(h)
        h = residual + h
        o = F.elu(h)

        return o

class ResidualModuleSingle(nn.Module):

    def __init__(self, inputSize):

        super(ResidualModuleSingle, self).__init__()

        self.conv1 = nn.Conv2d(inputSize, inputSize, 3, padding=1)
        self.conv2 = nn.Conv2d(inputSize, inputSize, 3, padding=1)

    def forward(self, inputData):

        h = F.elu(self.conv1(inputData))
        h = self.conv2(h)
        h = inputData + h
        o = F.elu(h)

        return o

class PathFinder(nn.Module):

    def __init__(self):

        super(PathFinder, self).__init__()

        self.flatShape = 1*1*64

        self.conv1 = nn.Conv2d(1, 8, 7, padding=4, stride=2)
        self.max1 = nn.MaxPool2d(2, 2)
        self.res1 = ResidualModuleDouble(8, 16)
        self.max2 = nn.MaxPool2d(2, 2)
        self.res2 = ResidualModuleDouble(16, 32)
        self.max3 = nn.MaxPool2d(2, 2)
        self.res3 = ResidualModuleDouble(32, 64)
        self.max4 = nn.MaxPool2d(2, 2)
        self.avg1 = nn.AvgPool2d(7, 7)
        self.lin1 = nn.Linear(self.flatShape, 8)

    def forward(self, boards):

        h = F.elu(self.conv1(boards))
        h = self.max1(h)
        h = self.res1(h)
        h = self.max2(h)
        h = self.res2(h)
        h = self.max3(h)
        h = self.res3(h)
        h = self.max4(h)
        h = self.avg1(h)
        h = h.view(-1, self.flatShape)
        h = self.lin1(h)
        o = torch.tanh(h)

        return o


