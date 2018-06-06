import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualModule(nn.Module):
    def __init__(self, inputSize, outputSize):

        super(ResidualModule, self).__init__()

        self.conv1 = nn.Conv2d(inputSize, inputSize, 3, padding=1)
        self.conv2 = nn.Conv2d(inputSize, outputSize, 3, padding=1)

    def forward(self, data):

        residual = torch.cat((data, data), 1)
        h = F.relu(self.conv1(data))
        h = self.conv2(h)
        h = residual + h
        o = F.relu(h)

        return o

class PathFinder(nn.Module):
    def __init__(self):

        super(PathFinder, self).__init__()

        self.flatShape = 14*14*256

        self.conv1 = nn.Conv2d(1, 64, 5, padding=2, stride=2)
        self.max1 = nn.MaxPool2d(2, 2)
        self.res1 = ResidualModule(64, 128)
        self.max2 = nn.MaxPool2d(2, 2)
        self.res2 = ResidualModule(128, 256)
        self.max3 = nn.MaxPool2d(2, 2)
        self.lin1 = nn.Linear(self.flatShape, 1024)
        self.lin2 = nn.Linear(1024, 1024)
        self.lin3 = nn.Linear(1024, 8)

    def forward(self, boards):

        h = F.relu(self.conv1(boards))
        h = self.max1(h)
        h = self.res1(h)
        h = self.max2(h)
        h = self.res2(h)
        h = self.max3(h)
        h = h.view(-1, self.flatShape)
        h = F.relu(self.lin1(h))
        h = F.relu(self.lin2(h))
        o = self.lin3(h)

        return o


