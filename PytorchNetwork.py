import torch.nn as nn
import torch.nn.functional as F

class PathFinder(nn.Module):
    def __init__(self):

        super(PathFinder, self).__init__()

        self.flatShape = 14*14*256

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.max1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.max2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.max3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.max4 = nn.MaxPool2d(2, 2)
        self.lin1 = nn.Linear(self.flatShape, 1024)
        self.lin2 = nn.Linear(1024, 1024)
        self.lin3 = nn.Linear(1024, 8)

    def forward(self, boards):

        h = F.relu(self.conv1(boards))
        h = self.max1(h)
        h = F.relu(self.conv2(h))
        h = self.max2(h)
        h = F.relu(self.conv3(h))
        h = self.max3(h)
        h = F.relu(self.conv4(h))
        h = self.max4(h)
        h = h.view(-1, self.flatShape)
        h = F.relu(self.lin1(h))
        h = F.relu(self.lin2(h))
        o = self.lin3(h)

        return o


