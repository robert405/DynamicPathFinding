import torch
from PytorchNetwork import PathFinder
from TrainingFunction import train, generateTrainingData
from EvaluationFunction import evaluateModel, showImgs, adjustBoardFromSim

saveIt = -1
savePath = "../SavedModel/pathFinderModel"

model = PathFinder().cuda()

nbUpdate = 25
batchSize = 50
lr = 1e-4
startRandTresh = 0.5 # start at 1 to have complete random choice in beginning


for i in range(25):

    trainSeq = generateTrainingData(model, nbUpdate, startRandTresh)

    boards = []
    rewards = []

    for board, move, reward in trainSeq:
        boards += [adjustBoardFromSim(board)]
        rewards += [reward]

    print(rewards)
    #showImgs(boards, 1, len(boards))


#evaluateModel(model)