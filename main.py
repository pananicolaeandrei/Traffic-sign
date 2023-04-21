
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import cv2
from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd
import random
from keras.preprocessing.image import ImageDataGenerator


path = "myData"
labelFile = 'labels.csv'
batch_size_val = 50
steps_per_epoch_val = 2000
epochs_val = 20
imageDimesions = (32, 32, 3)
testRatio = 0.2
validationRatio = 0.2  