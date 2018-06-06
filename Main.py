import torch
from PytorchNetwork import PathFinder
from TrainingFunction import train
from EvaluationFunction import evaluateModel
import matplotlib.pyplot as plt

def showImgs(imgs, nbEx, nbCl):
    counter = 0
    for i in range(nbCl):
        for j in range(nbEx):
            plt.subplot(nbEx, nbCl, counter+1)
            plt.imshow(imgs[counter].astype('uint8'))
            plt.axis('off')
            counter += 1

    plt.show()

saveIt = 0
savePath = "../SavedModel/pathFinderModel"

model = PathFinder().cuda()
if (saveIt >= 0):
    model.load_state_dict(torch.load(savePath + str(saveIt)))

nbIteration = 500
nbUpdate = 20
batchSize = 60
lr = 1e-4
startRandTresh = 1 # must start at 1 to have complete random choice in beginning
randTreshRate = 75

lossList = train(model, nbIteration, nbUpdate, batchSize, lr, startRandTresh, randTreshRate)

torch.save(model.state_dict(), savePath + str(saveIt + 1))

plt.plot(lossList)
plt.show()

evaluateModel(model)