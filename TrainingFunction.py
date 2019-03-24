import torch
import torch.nn as nn
import numpy as np
import time
from SimWorld.Engine import Engine
from random import shuffle

def train(model, data, nbIteration, batchSize, lr):

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    lossList = []
    model.train()
    start = time.time()

    for k in range(nbIteration):

        shuffle(data)
        count = 0
        meanLoss = 0

        for i in range(0,len(data),batchSize):

            batch = data[i:i+batchSize]
            boards = []
            moves = []
            rewards = []

            for board, move, reward in batch:

                boards += [board]
                moves += [move]
                rewards += [reward]

            boards = torch.FloatTensor(boards).cuda()
            boards = torch.unsqueeze(boards, 1)
            index = torch.arange(len(moves)).cuda()
            moves = torch.LongTensor(moves).cuda()
            rewards = torch.FloatTensor(rewards).cuda()

            allPred = model(boards)
            allPred = allPred[index,moves]

            loss = criterion(allPred, rewards)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            currentLoss = loss.data.cpu().numpy()
            meanLoss += currentLoss
            count += 1

        meanLoss = meanLoss / count
        lossList += [meanLoss]

    end = time.time()
    print("Training time in second : " + str(end - start))

    return lossList


def generateTrainingData(model, nbUpdate, randTresh, forget):

    model.eval()
    engine = Engine(1, (15, 15), (15, 15), 224)
    trainSeq = []

    for i in range(nbUpdate):

        boards = engine.drawAllBoard()
        move = np.random.randint(0, high=8)

        if (randTresh < np.random.random()):

            torchBoards = torch.FloatTensor(boards).cuda()
            torchBoards = torch.unsqueeze(torchBoards, 1)
            allPred = model(torchBoards)
            target = allPred.data.cpu().numpy()
            allMove = np.argmax(target, axis=1)
            move = allMove[0]

        engine.update([move])
        newRobotPos = engine.getAllRobotPos()
        penalty = engine.calculateObstaclePenalty(newRobotPos)
        finalReward = engine.calculateFinalReward(newRobotPos)

        if (penalty > 0):
            trainSeq += [(boards[0], move, -1)]
            break

        if (finalReward > 0):

            reward = 1
            rewardTrainSeq = [(boards[0], move, reward)]

            for j in range(i-1,-1,-1):

                board, oldMove, _ = trainSeq[j]
                reward = reward * forget
                rewardTrainSeq += [(board, oldMove, reward)]

            trainSeq = rewardTrainSeq
            break

        trainSeq += [(boards[0], move, 0)]

    return trainSeq





