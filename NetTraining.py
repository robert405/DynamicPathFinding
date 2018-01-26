from NetUtils import *
import time
from Engine import Engine
from Utils import showImgs
from math import isnan

sess = tf.Session()

y = tf.placeholder(tf.float32, shape=[None, 9], name = "y")
x = tf.placeholder(tf.float32, shape=[None, 224,224,1], name = "x")
x_norm = tf.divide(tf.subtract(x, tf.to_float(128)), tf.to_float(128), name="imgNorm")

#-------------------------------------------------------------

h_conv = pathFindingNet(x_norm)

description = tf.identity(h_conv, name="descriptor")

h_avPool = tf.layers.average_pooling2d(description, [7, 7], [7, 7], name="Avg_Pooling")
inputSize = 1 * 1 * 512
h_avPool_flat = tf.reshape(h_avPool, [-1, inputSize], name = "Flatening")

pred = fullyLayerNoRelu("1", h_avPool_flat, 512, 9)
softmax = tf.nn.softmax(pred,name="Softmax")
move = tf.argmax(softmax,axis=1,name="Move")

learning_rate = tf.placeholder(tf.float32, shape=[], name = "Learning_Rate")

with tf.name_scope("TrainStep"):

    epsilonSoftmax = softmax + 2.220446049250313e-16
    loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(epsilonSoftmax), 1), name="Loss")
    train_step = tf.train.AdamOptimizer(learning_rate,name="Training").minimize(loss)

tf.summary.scalar('Compute-Loss',loss)

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./result", sess.graph)

init = tf.global_variables_initializer()
tvars = tf.trainable_variables()
saver = tf.train.Saver(tvars)
sess.run(init)

lr = 1e-4

globalStartTime = time.time()

nbUpdate = 15
batchSize = 50
counter = 1
nbSimulation = 10000

for k in range(nbSimulation):

    print("==================================================")
    print("Doing simulation no " + str(k))
    print("Learning rate : " + str(lr))

    localStartTime = time.time()
    engine = Engine(batchSize,(15,15),(15,15),224,1)

    lossMean = 0

    for i in range(nbUpdate):

        boards = engine.getAllBoardForNet()
        oldRobotPos = engine.getAllCurrentRobotPos()

        allMove, allProb = sess.run([move, softmax],feed_dict={x:boards})
        newRobotPos = engine.getAllFuturRobotPos(allMove)

        reward = engine.calculateReward(oldRobotPos, newRobotPos)
        targetLabel = allProb
        rowIndexing = np.arange(batchSize)
        targetLabel[rowIndexing,allMove] = reward

        if (i == 0):
            summary,_,calculatedLoss = sess.run([merged,train_step,loss],feed_dict={x:boards, y:targetLabel, learning_rate:lr})
            lossMean += calculatedLoss
            counter += 1
            writer.add_summary(summary,counter)
        else:
            _,calculatedLoss = sess.run([train_step,loss],feed_dict={x:boards,y:targetLabel,learning_rate:lr})
            lossMean += calculatedLoss

        engine.update(allMove)


    mean = lossMean/nbUpdate
    print("Mean Loss : "+str(mean))

    if (k == 8000):
        lr = lr*0.1

    localEndTime = time.time()
    localElapsedTime = localEndTime - localStartTime
    print("Local elapsed time (sec) : " + str(localElapsedTime))
    print("==================================================")

    if (mean < 0.01 or isnan(mean)):
        break


globalEndTime = time.time()
globalElapsedTime = globalEndTime - globalStartTime

print("Global elapsed time (min): " + str(globalElapsedTime/60))
print("Nb update per simulation : " + str(nbUpdate))
print("Nb simulation done : " + str(nbSimulation))

saver.save(sess, "./model/model.ckpt")


while(True):

    allBoard = []
    engine = Engine(1,(15,15),(15,15),224,1)
    boards = engine.drawAllBoard()
    allBoard += [boards[0]]

    for i in range(24):

        netBoards = engine.getAllBoardForNet()
        allMove = sess.run([move],feed_dict={x:netBoards})
        engine.update(allMove[0])

        boards = engine.drawAllBoard()
        allBoard += [boards[0]]

    showImgs(allBoard, 5, 5)

    print("exit or continu?")
    answer = input()

    if (answer == "exit"):
        break

















