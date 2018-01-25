from Engine import Engine
import numpy as np
from Utils import showImgs

engine = Engine(5, (10,10), (10,10), 224, 1)

boards = engine.drawAllBoard()
oldRobotPos = engine.getAllCurrentRobotPos()
print(oldRobotPos)
showImgs(boards,1,5)

penaltyBoards = engine.createAllPenaltyBoard()
showImgs(penaltyBoards,1,5)

robotMove = np.ones((5),dtype="Int32")
engine.update(robotMove)
newRobotPos = engine.getAllCurrentRobotPos()

boards = engine.drawAllBoard()
print(newRobotPos)
showImgs(boards,1,5)

reward = engine.calculateReward(oldRobotPos, newRobotPos)
print(reward)

penaltyBoards = engine.createAllPenaltyBoard()
showImgs(penaltyBoards,1,5)

boardForNet = engine.getAllBoardForNet()

print(boardForNet.shape)

