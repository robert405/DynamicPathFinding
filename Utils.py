import matplotlib.pyplot as plt
import pickle
from random import sample, shuffle

def adjustBoardFromSim(board):

    return (board + 1.0) * 127.5

def balanceDataset(posData, negData):

    nbPosData = len(posData)
    nbNegData = len(negData)

    minData = min(nbNegData, nbPosData)

    newPosData = sample(posData, minData)
    newNegData = sample(negData, minData)

    balanceData = newNegData + newPosData
    shuffle(balanceData)

    return balanceData

def sepInPosNegAndRemoveZero(data):

    posData = []
    negData = []

    for board, move, reward in data:

        if (reward < 0):
            negData += [(board, move, reward)]
        elif (reward > 0):
            posData += [(board, move, reward)]

    return posData, negData


def showImgs(imgs, nbEx, nbCl, title=None, savePath=None, removeAxis=True, size=(8,8)):

    counter = 0
    fig = plt.figure(figsize=size)
    for i in range(nbCl):
        for j in range(nbEx):
            plt.subplot(nbEx, nbCl, counter+1)
            plt.imshow(imgs[counter].astype('uint8'))
            if (removeAxis):
                plt.axis('off')
            counter += 1

    if (title is not None):
        fig.suptitle(title, fontsize=16)

    if (savePath is not None):
        fig.savefig(savePath)
    else:
        plt.show()

    plt.close(fig)


def plotGraph(title, xData, yData, labels, ax1Name, ax2Name, figSize=(8, 8), xLim=None, yLim=None, savePath=None):

    fig = plt.figure(figsize=figSize)
    figName = title
    fig.suptitle(figName, fontsize=20)

    graph = plt.subplot('111')

    for i in range(len(xData)):
        graph.plot(xData[i], yData[i], label=labels[i])

    if (xLim is not None):
        graph.set_xlim(xLim)

    if (yLim is not None):
        graph.set_ylim(yLim)

    box = graph.get_position()
    graph.set_position([box.x0, box.y0 - box.height*0.02, box.width, box.height])
    graph.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=len(labels))

    plt.xlabel(ax1Name, fontsize=10)
    plt.ylabel(ax2Name, fontsize=10)

    if (savePath is not None):
        fig.savefig(savePath)
    else:
        plt.show()

    plt.close(fig)


def saveData(data, savePath):

    with open(savePath, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def loadData(savePath):

    with open(savePath, 'rb') as handle:
        data = pickle.load(handle)

    return data








