from PytorchNetwork import PathFinder
from TrainingFunction import train, generateTrainingData
from EvaluationFunction import evaluateModel
from Utils import plotGraph, balanceDataset, sepInPosNegAndRemoveZero

model = PathFinder().cuda()

nbGenSeq = 500
maxNbUpdate = 25
randTresh = 1 # start at 1 to have complete random choice in beginning
forget = 0.95

nbIteration = 15
batchSize = 60
lr = 5e-4

posDataset = []
negDataset = []

for k in range(5):

    print("Doing : " + str(k+1) + "/" + str(5))

    print("Generating Data")
    print("randTresh : " + str(randTresh))
    trainData = []
    for i in range(nbGenSeq):
        trainData += generateTrainingData(model, maxNbUpdate, randTresh, forget)

    posData, negData = sepInPosNegAndRemoveZero(trainData)
    posDataset += posData
    negDataset += negData

    print("nb Pos Data : " + str(len(posData)))
    print("nb Neg Data : " + str(len(negData)))

    print("Total nb Pos Data : " + str(len(posDataset)))
    print("Total nb Neg Data : " + str(len(negDataset)))

    balDataset = balanceDataset(posDataset, negDataset)
    randTresh = randTresh * 0.6

    print("Starting trainning!")
    lossList = train(model, balDataset, nbIteration, batchSize, lr)

    title = "Loss by iteration"
    labels = ["train loss"]
    ax1Name = "Iteration"
    ax2Name = "Loss"

    plotGraph(title, [list(range(len(lossList)))], [lossList], labels, ax1Name, ax2Name)

    evaluateModel(model)

    print("-------------------------------------------------------------------------")