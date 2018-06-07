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
        h = F.leaky_relu(self.conv1(inputData), 1e-2)
        h = self.conv2(h)
        h = residual + h
        o = F.leaky_relu(h, 1e-2)

        return o

class ResidualModuleSingle(nn.Module):
    def __init__(self, inputSize):

        super(ResidualModuleSingle, self).__init__()

        self.conv1 = nn.Conv2d(inputSize, inputSize, 3, padding=1)
        self.conv2 = nn.Conv2d(inputSize, inputSize, 3, padding=1)

    def forward(self, inputData):

        h = F.leaky_relu(self.conv1(inputData), 1e-2)
        h = self.conv2(h)
        h = inputData + h
        o = F.leaky_relu(h, 1e-2)

        return o

class PathFinder(nn.Module):
    def __init__(self):

        super(PathFinder, self).__init__()

        self.flatShape = 1*1*512

        self.conv1 = nn.Conv2d(1, 64, 7, padding=4, stride=2)
        self.max1 = nn.MaxPool2d(2, 2)
        self.res1 = ResidualModuleDouble(64, 128)
        self.max2 = nn.MaxPool2d(2, 2)
        self.res2 = ResidualModuleDouble(128, 256)
        self.max3 = nn.MaxPool2d(2, 2)
        self.res3 = ResidualModuleDouble(256, 512)
        self.max4 = nn.MaxPool2d(2, 2)
        self.res4 = ResidualModuleSingle(512)
        self.avg1 = nn.AvgPool2d(7, 7)
        self.lin1 = nn.Linear(self.flatShape, 8)

    def forward(self, boards):

        h = F.leaky_relu(self.conv1(boards), 1e-2)
        h = self.max1(h)
        h = self.res1(h)
        h = self.max2(h)
        h = self.res2(h)
        h = self.max3(h)
        h = self.res3(h)
        h = self.max4(h)
        h = self.res4(h)
        h = self.avg1(h)
        h = h.view(-1, self.flatShape)
        o = self.lin1(h)

        return o


