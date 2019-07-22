"""
Description: Train emotion classification model
"""
import matplotlib.pyplot as plt
from confusionX_roc import *
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from load_and_process import load_fer2013
from load_and_process import preprocess_input
from sklearn.model_selection import train_test_split
from mul_ksize_cnn import *
from mobilenet import *
from mobilenet_v2 import *
import resnet
from keras.models import load_model
import os, glob
import cv2
from keras.utils import to_categorical
import numpy as np
from load_retrain_data import *

##################### 加载模型 ###################################################################
load_model_path = 'models/best_model/MUL_KSIZE_MobileNet_v2_best.hdf5'
model = load_model(load_model_path, custom_objects={'tf': tf, 'relu6': relu6})
model.summary()

###################### loading fer2013 dataset ###############################################
faces, emotions = load_fer2013()
faces = preprocess_input(faces)
x_train, x_test, y_train, y_test = train_test_split(faces, emotions,test_size=0.2, shuffle=False)
x_PublicTest, x_PrivateTest, y_PublicTest, y_PrivateTest = train_test_split(x_test, y_test,test_size=0.5,shuffle=False)

######################## 画出测试集混淆矩阵 ##############################################
pred = model.predict(x_test)

PrivateTest_score = model.evaluate(x_test, y_test)
print('PublicTest accuracy:', PrivateTest_score[1])

# labels_name = ["angry" ,"disgust","fear", "happy", "sad", "surprised"]
labels_name = ["Ang.", "Dis.", "Fear", "Hap.", "Sad", "Sur.", "Neu."]
# labels_name = ["Ang.", "Dis.", "Fear", "Hap.", "Sad", "Sur."]

paintConfusion_float(ture_labels=y_test, pred_labels=pred,
                     labels_name=labels_name,
                     # save_path='./models/pictures/fer2013_PrivateTest1_{0:.3f}'.format(PrivateTest_score[1])+'.png'
                     )
