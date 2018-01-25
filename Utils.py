import pickle
import numpy as np
import scipy.ndimage as ndimage
from PIL import Image
import matplotlib.pyplot as plt
from random import random


# Various constants for the size of the images.
# Use these constants in your own program.


def unpickle(file):
    with open(file,'rb') as fo:
        dict = pickle.load(fo,encoding='bytes')

    return dict

def convert_images(raw):

    img_size = 32
    num_channels = 3
    # Convert the raw images from the data-files to floating-points.
    raw_int = np.array(raw,dtype=int)

    # Reshape the array to 4-dimensions.
    images = raw_int.reshape([-1,num_channels,img_size,img_size])

    # Reorder the indices of the array.
    images = images.transpose([0,2,3,1])

    return images


def load_data(filename):
    # Load the pickled data-file.
    data = unpickle(filename)

    # Get the raw images.
    raw_images = data[b'data']

    # Get the class-numbers for each image. Convert to numpy-array.
    cifarClass = np.array(data[b'fine_labels'])

    # Convert the images.
    images = convert_images(raw_images)

    return images,cifarClass

def showImgs(imgs, nbEx, nbCl):
    counter = 0
    for i in range(nbCl):
        for j in range(nbEx):
            plt.subplot(nbEx, nbCl, counter+1)
            plt.imshow(imgs[counter].astype('uint8'))
            plt.axis('off')
            counter += 1

    plt.show()

def getCropOfMatrix(img, left, top, shape):

    right = left + shape[0]
    bottom = top + shape[1]
    return img[left:right, top:bottom]

def shiftV(img, nb):

    m,n,o = img.shape

    copyCrop = getCropOfMatrix(img, 0, 0, (nb, n))
    flipCrop = np.flipud(copyCrop)
    imgCrop = getCropOfMatrix(img, 0, 0, ((m - nb), n))
    shiftImg = np.concatenate((flipCrop,imgCrop), axis=0)

    return shiftImg


def brighten(imgs):

    temp = imgs * 1.35
    temp[temp > 255] = 255

    return temp

def darken(imgs):

    return imgs * 0.65

def blur(img):

    return ndimage.gaussian_filter(img, sigma=(1, 1, 0), order=0)

def dataAugmentationOld(data, label):

    index = 0
    initialShape = data.shape
    newLeftRight = np.empty(shape=initialShape)
    #newUpDown = np.empty(shape=initialShape)
    #newLeftRightUpDown = np.empty(shape=initialShape)
    #newRot90 = np.empty(shape=initialShape)
    #newRot270 = np.empty(shape=initialShape)
    #newLeftRightRot90 = np.empty(shape=initialShape)
    #newLeftRightRot270 = np.empty(shape=initialShape)
    newBlur = np.empty(shape=initialShape)
    newBrighten = brighten(data)
    newDarken = darken(data)

    for innerArray in data:
        leftRight = np.fliplr(innerArray)
        newLeftRight[index] = leftRight
        #newUpDown[index] = np.flipud(innerArray)
        #newLeftRightUpDown[index] = np.flipud(leftRight)
        #newRot90[index] = np.rot90(innerArray)
        #newRot270[index] = np.rot90(innerArray, 3)
        #newLeftRightRot90[index] = np.rot90(leftRight)
        #newLeftRightRot270[index] = np.rot90(leftRight,3)
        newBlur[index] = blur(innerArray)
        index += 1

    newLabel = label
    for i in range(4):
        newLabel = np.concatenate((newLabel,label),axis=0)

    newData = np.concatenate((data,newLeftRight),axis=0)
    #newData = np.concatenate((newData,newUpDown),axis=0)
    #newData = np.concatenate((newData,newLeftRightUpDown),axis=0)
    #newData = np.concatenate((newData,newRot90),axis=0)
    #newData = np.concatenate((newData,newRot270),axis=0)
    #newData = np.concatenate((newData,newLeftRightRot90),axis=0)
    #newData = np.concatenate((newData,newLeftRightRot270),axis=0)
    newData = np.concatenate((newData,newBlur),axis=0)
    newData = np.concatenate((newData,newBrighten),axis=0)
    newData = np.concatenate((newData,newDarken),axis=0)

    return newData, newLabel


def dataAugmentation(img, label):

    newLeftRight = img[...,::-1,:]
    newBlur = ndimage.gaussian_filter(img, sigma=(0, 1, 1, 0), order=0)

    imgs = np.concatenate((img,newLeftRight), axis=0)
    imgs = np.concatenate((imgs,newBlur),axis=0)

    labels = np.concatenate((label,label),axis=0)
    labels = np.concatenate((labels,label),axis=0)

    return imgs, labels

def dataAugmentationLarge(img, label):

    modif1 = img
    modif2 = img
    newLeftRight = img[...,::-1,:]

    if (random() < 0.25):
        modif1 = newLeftRight
        modif2 = newLeftRight
    elif(random() < 0.5):
        modif1 = newLeftRight
    elif(random() < 0.75):
        modif2 = newLeftRight

    newBrighten = brighten(modif1)
    newDarken = darken(modif2)
    newBlur = ndimage.gaussian_filter(img, sigma=(0, 1, 1, 0), order=0)

    imgs = np.concatenate((img,newLeftRight), axis=0)
    imgs = np.concatenate((imgs,newBrighten),axis=0)
    imgs = np.concatenate((imgs,newDarken),axis=0)
    imgs = np.concatenate((imgs,newBlur),axis=0)

    labels = np.concatenate((label,label),axis=0)
    labels = np.concatenate((labels,label),axis=0)
    labels = np.concatenate((labels,label),axis=0)
    labels = np.concatenate((labels,label),axis=0)

    return imgs, labels


def dataAugmentationForNormalisation(img, label):

    newLeftRight = np.fliplr(img)
    newBrighten = brighten(img)
    newDarken = darken(newLeftRight)
    newBlur = blur(img)

    data = [img,newBrighten,newDarken,newLeftRight,newBlur]
    newData = []

    for pic in data:
        newData += [np.array([pic,label])]

    return newData


def resize(img, width, height):
    img = np.atleast_3d(img)
    imgHeigth, imgWidth, channel = img.shape

    if (channel == 1):
        tempImg = np.concatenate((img,img),axis=2)
        img = np.concatenate((tempImg,img),axis=2)
        channel = 3

    if (imgWidth != imgHeigth):
        if (imgWidth < imgHeigth):
            padLeftSize, padRigthSize = calculatePaddingSize(imgHeigth, imgWidth)
            padLeft = np.zeros((imgHeigth,padLeftSize,channel))
            img = np.concatenate((padLeft,img),axis=1)
            padRigth = np.zeros((imgHeigth,padRigthSize,channel))
            img = np.concatenate((img,padRigth),axis=1)
        else:
            padTopSize, padBottomSize = calculatePaddingSize(imgWidth, imgHeigth)
            padTop = np.zeros((padTopSize,imgWidth,channel))
            img = np.concatenate((padTop,img),axis=0)
            padBottom = np.zeros((padBottomSize,imgWidth,channel))
            img = np.concatenate((img,padBottom),axis=0)

    # resized
    pilImg = Image.fromarray(np.uint8(img))
    pilImg = pilImg.resize((width, height), Image.ANTIALIAS)

    return np.asarray(pilImg, dtype="int32" )

def calculatePaddingSize(desiredValue, valueToChange):

    diff = desiredValue - valueToChange
    padOne = int(diff / 2)
    padTwo = diff - padOne

    return padOne, padTwo


def loadImage(infilename) :
    img = Image.open(infilename)
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

