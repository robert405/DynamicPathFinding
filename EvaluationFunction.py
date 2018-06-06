import torch
import numpy as np
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

def evaluateModel(model):

    model.eval()
    with torch.no_grad():
        while (True):

            allBoard = []
            engine = Engine(1, (15, 15), (15, 15), 224)
            boards = engine.drawAllBoard()
            board = boards[0] + 2.0
            board = board * (252 / 4)
            allBoard += [board]

            for i in range(24):
                netBoards = engine.getAllBoardForNet()
                torchBoards = torch.FloatTensor(netBoards).cuda()
                torchBoards = torchBoards.transpose(1, 3)
                predMove = model(torchBoards)
                allMove = predMove.data.cpu().numpy()
                allMove = np.argmax(allMove, axis=1)
                engine.update(allMove)

                boards = engine.drawAllBoard()
                board = boards[0] + 2.0
                board = board * (252 / 4)
                allBoard += [board]

            showImgs(allBoard, 5, 5)

            print("exit or continu?")
            answer = input()

            if (answer == "exit"):
                break