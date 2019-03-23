import torch
from PytorchNetwork import PathFinder
from TrainingFunction import train
from EvaluationFunction import evaluateModel
import matplotlib.pyplot as plt

saveIt = -1
savePath = "../SavedModel/pathFinderModel"

model = PathFinder().cuda()
#if (saveIt >= 0):
#    model.load_state_dict(torch.load(savePath + str(saveIt)))

nbIteration = 10000
nbUpdate = 1
batchSize = 50
lr = 1e-4
startRandTresh = 1 # start at 1 to have complete random choice in beginning
randTreshRate = 100000

#lossList = train(model, nbIteration, nbUpdate, batchSize, lr, startRandTresh, randTreshRate)

#torch.save(model.state_dict(), savePath + str(saveIt + 1))

#plt.plot(lossList)
#plt.show()

evaluateModel(model)