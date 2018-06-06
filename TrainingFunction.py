import torch
import torch.nn as nn
import numpy as np
import time
from Engine import Engine

def train(model, nbIteration, nbUpdate, batchSize, lr, startRandTresh, randTreshRate):

    print("Starting trainning!")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    randMoveTreshold = startRandTresh
    lossList = []
    moduloPrint = 50
    model.train()
    start = time.time()

    for k in range(nbIteration):

        meanLoss = 0
        engine = Engine(batchSize,(15,15),(15,15),224)
        trainSeq = []
        newRobotPos = np.empty(batchSize)

        for i in range(nbUpdate):

            boards = engine.getAllBoardForNet()
            oldRobotPos = engine.getAllRobotPos()
            allMove = np.random.rand(batchSize) * 8
            allMove = np.floor(allMove).astype('int')

            if (randMoveTreshold < np.random.random()):
                with torch.no_grad():
                    torchBoards = torch.FloatTensor(boards).cuda()
                    torchBoards = torchBoards.transpose(1, 3)
                    allPred = model(torchBoards)
                    target = allPred.data.cpu().numpy()
                    allMove = np.argmax(target, axis=1)

            engine.update(allMove)
            newRobotPos = engine.getAllRobotPos()
            stepReward = engine.calculateStepReward(oldRobotPos, newRobotPos)

            trainSeq += [(boards, allMove, stepReward)]

        finalReward = engine.calculateFinalReward(newRobotPos)
        currentReward = finalReward

        for i in range((nbUpdate-1),-1,-1):

            boards, allMove, reward = trainSeq[i]
            torchBoards = torch.FloatTensor(boards).cuda()
            torchBoards = torchBoards.transpose(1, 3)
            allPred = model(torchBoards)
            currentReward = reward + (0.9 * currentReward)
            currentReward = np.maximum(currentReward, -1)
            rowIndexing = np.arange(batchSize)
            target = allPred.data.cpu().numpy()
            target[rowIndexing, allMove] = currentReward
            torchTarget = torch.FloatTensor(target).cuda()

            loss = criterion(allPred, torchTarget)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            currentLoss = loss.data.cpu().numpy()
            meanLoss += currentLoss
            lossList += [currentLoss]

        if (k % moduloPrint == 0):
            print("Iteration : " + str(k+1) + " / " + str(nbIteration) + ", Current mean loss : " + str(meanLoss/(moduloPrint*nbUpdate)))

        if ((k+1) % randTreshRate == 0 and randMoveTreshold > 0):
            randMoveTreshold += -0.05

        if (k % 200 == 0):
            end = time.time()
            timeTillNow = end - start
            predictedRemainingTime = (timeTillNow / (k + 1)) * (nbIteration - (k + 1))
            print("--------------------------------------------------------------------")
            print("Time to run since started (sec) : " + str(timeTillNow))
            print("Predicted remaining time (sec) : " + str(predictedRemainingTime))
            print("--------------------------------------------------------------------")


    end = time.time()
    print("Time to run in second : " + str(end - start))

    return lossList