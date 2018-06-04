import torch
import numpy as np
from PytorchNetwork import PathFinder
from TrainingFunction import train
from Engine import Engine
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


savePath = '../SavedModel/pathFinderModel.pt'

model = PathFinder().cuda()
model.load_state_dict(torch.load(savePath))

nbIteration = 1000
nbUpdate = 25
batchSize = 60
lr = 1e-3
startRandTresh = 0.2
randTreshRate = 100

lossList = train(model, nbIteration, nbUpdate, batchSize, lr, startRandTresh, randTreshRate)

torch.save(model.state_dict(), savePath)

plt.plot(lossList)
plt.show()

model.eval()
with torch.no_grad():
    while(True):

        allBoard = []
        engine = Engine(1,(15,15),(15,15),224)
        boards = engine.drawAllBoard()
        allBoard += [boards[0]]

        for i in range(24):

            netBoards = engine.getAllBoardForNet()
            torchBoards = torch.FloatTensor(netBoards).cuda()
            torchBoards = torchBoards.transpose(1, 3)
            predMove = model(torchBoards)
            allMove = predMove.data.cpu().numpy()
            allMove = np.argmax(allMove, axis=1)
            engine.update(allMove)

            boards = engine.drawAllBoard()
            allBoard += [boards[0]]

        showImgs(allBoard, 5, 5)

        print("exit or continu?")
        answer = input()

        if (answer == "exit"):
            break