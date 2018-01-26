from Simulator import Simulation
import numpy as np

class Engine:

    def __init__(self, nbSimulation, robotShape, goalShape, boardSize, nbChannel):

        self.boardSize = boardSize
        self.nbChannel = nbChannel
        self.robotShape = robotShape
        self.goalShape = goalShape
        self.robotShape = robotShape
        self.nbSimulation = nbSimulation
        self.simulationList = []
        self.allRobotPos = np.zeros((self.nbSimulation,2),dtype=np.int32)
        self.allGoalPos = np.zeros((self.nbSimulation,2),dtype=np.int32)
        self.actionMove = self.createActionMoveAssociation()
        self.quadrant = self.createPossiblePosition()
        self.createSimulation()

    def createSimulation(self):

        self.simulationList = []
        self.allRobotPos = np.zeros((self.nbSimulation,2),dtype=np.int32)
        self.allGoalPos = np.zeros((self.nbSimulation,2),dtype=np.int32)

        for i in range(self.nbSimulation):

            nb1 = np.random.randint(0,high=(4))
            nb2 = np.random.randint(0,high=(4))
            while (nb2 == nb1):
                nb2 = np.random.randint(0,high=(4))

            quadrant1 = self.quadrant[nb1]
            quadrant2 = self.quadrant[nb2]

            xRobotPos = np.random.randint(quadrant1[0][0],high=(quadrant1[0][1] - self.robotShape[0]))
            yRobotPos = np.random.randint(quadrant1[1][0],high=(quadrant1[1][1] - self.robotShape[1]))
            xGoalPos = np.random.randint(quadrant2[0][0],high=(quadrant2[0][1] - self.goalShape[0]))
            yGoalPos = np.random.randint(quadrant2[1][0],high=(quadrant2[1][1] - self.goalShape[1]))

            self.allRobotPos[i,0] = xRobotPos
            self.allRobotPos[i,1] = yRobotPos
            self.allGoalPos[i,0] = xGoalPos
            self.allGoalPos[i,1] = yGoalPos

            self.simulationList += [Simulation((xRobotPos,yRobotPos), self.robotShape, (xGoalPos,yGoalPos), self.goalShape, self.boardSize, self.nbChannel)]

    def update(self, robotMoveList):

        for i in range(self.nbSimulation):
            currentMove = self.actionMove[robotMoveList[i]]
            currentMove = self.checkInLimit(self.allRobotPos[i],currentMove)
            self.allRobotPos[i] = self.allRobotPos[i] + currentMove
            self.simulationList[i].update(currentMove)

    def getAllCurrentRobotPos(self):

        return np.copy(self.allRobotPos)

    def getAllFuturRobotPos(self, robotMoveList):

        futurPos = np.zeros_like(self.allRobotPos)

        for i in range(self.nbSimulation):

            currentMove = self.actionMove[robotMoveList[i]]
            currentMove = self.checkInLimit(self.allRobotPos[i], currentMove)
            futurPos[i] = self.allRobotPos[i] + currentMove

        return futurPos

    def checkInLimit(self, pos, move):

        if (not (pos[0] + move[0] < self.boardSize - self.robotShape[0])):
            move[0] = 0

        elif (not (0 < pos[0] + move[0])):
            move[0] = 0

        if (not (pos[1] + move[1] < self.boardSize-self.robotShape[1])):
            move[1] = 0

        elif (not (0 < pos[1] + move[1])):
            move[1] = 0

        return move



    def getAllGoalPos(self):

        return np.copy(self.allGoalPos)

    def drawAllBoard(self):

        boardList = []

        for simulation in self.simulationList:

            boardList += [simulation.drawBoard()]

        return boardList

    def getAllBoardForNet(self):

        boardList = np.zeros((self.nbSimulation,self.boardSize,self.boardSize,1))

        for i in range(self.nbSimulation):
            boardList[i] = self.simulationList[i].getBoardForNet()

        return boardList

    def createAllPenaltyBoard(self):

        boardList = []

        for simulation in self.simulationList:

            boardList += [simulation.createPenaltyBoard()]

        return boardList

    def calculateReward(self, oldRobotPos, newRobotPos):

        oldDist = self.getDist(oldRobotPos)
        newDist = self.getDist(newRobotPos)
        penaltyList = self.calculateObstaclePenalty(newRobotPos)

        diff = oldDist - newDist
        diff = np.maximum(diff,np.zeros_like(diff))
        norm = diff / 6
        norm = np.minimum(norm,np.ones_like(norm))

        reward = norm - penaltyList
        reward = np.maximum(reward,np.zeros_like(reward))

        return reward

    def getDist(self,robotPos):

        diff = self.allGoalPos - robotPos
        power = diff**2
        dist = np.sqrt(np.sum(power,axis=1))

        return dist

    def calculateObstaclePenalty(self, newRobotPos):

        penaltyList = np.zeros((self.nbSimulation))

        for i in range(self.nbSimulation):

            penaltyBoard = self.simulationList[i].createPenaltyBoard()
            allPenalty = penaltyBoard[newRobotPos[i,0]:newRobotPos[i,0] + self.robotShape[0], newRobotPos[i,1]:newRobotPos[i,1] + self.robotShape[1]]
            penaltyList[i] = np.sum(allPenalty)

        return penaltyList

    def createActionMoveAssociation(self):

        actionMoveDict = {
            0:np.array([0,0]),
            1:np.array([6,0]),
            2:np.array([0,6]),
            3:np.array([-6,0]),
            4:np.array([0,-6]),
            5:np.array([6,6]),
            6:np.array([-6,-6]),
            7:np.array([6,-6]),
            8:np.array([-6,6])
        }

        return actionMoveDict

    def createPossiblePosition(self):

        half = int(self.boardSize / 2)
        halfMinus = half - 25
        halfPlus = half + 25

        leftTop = (0,halfMinus)
        rigthBottom = (halfPlus,self.boardSize-1)

        possiblePos = {
            0:(leftTop,leftTop),
            1:(leftTop,rigthBottom),
            2:(rigthBottom,rigthBottom),
            3:(rigthBottom,leftTop)
        }

        return possiblePos