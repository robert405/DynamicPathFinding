import torch
import numpy as np
from SimWorld.Engine import Engine
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

def adjustBoardFromSim(board):

    return (board + 1.0) * 127.5

def evaluateModel(model):

    model.eval()
    with torch.no_grad():
        while (True):

            allBoard = []
            engine = Engine(1, (15, 15), (15, 15), 224)
            boards = engine.drawAllBoard()
            allBoard += [adjustBoardFromSim(boards[0])]

            for i in range(24):
                boards = engine.drawAllBoard()
                torchBoards = torch.FloatTensor(boards).cuda()
                torchBoards = torch.unsqueeze(torchBoards, 1)
                predMove = model(torchBoards)
                allMove = predMove.data.cpu().numpy()
                allMove = np.argmax(allMove, axis=1)
                engine.update(allMove)

                allBoard += [adjustBoardFromSim(boards[0])]

            showImgs(allBoard, 5, 5)

            print("exit or continu?")
            answer = input()

            if (answer == "exit"):
                break