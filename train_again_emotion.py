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
# from models.cnn import *
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

# parameters
batch_size = 16
num_epochs = 1000
input_shape = (48, 48, 1)
# input_shape = (181, 143, 3)
verbose = 1
num_classes = 6
patience = 30
##################### 读取数据集路径 ###################################################################
retrain_data_path = 'other_dataset'

##################### 加载模型路径 ###################################################################
load_model_path = 'models/best_model/MUL_KSIZE_MobileNet_v2_best.hdf5'
##################### 模型保存路径和名字 ###################################################################
base_path = 'models/'
running_model_name = 'MUL_KSIZE_MobileNet_v2_best_ck+'
trained_models_path = base_path + running_model_name
TensorBoard_logdir_path = './models/log/MUL_KSIZE_MobileNet_v2_best_ck+'


# data generator
data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True,
                        fill_mode='nearest')

pre_model = load_model(load_model_path, custom_objects={'tf': tf, 'relu6': relu6})
# pre_model = MUL_KSIZE_MobileNet_v2_best(input_shape=input_shape, num_classes=num_classes)
# pre_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# pre_model.summary()

########################## re-build network ################################################
pooling_output = pre_model.get_layer('global_average_pooling2d_1').output
x = Reshape((1, 1, 1280))(pooling_output)
x = Dropout(0.3, name='Dropout')(x)
x = Conv2D(num_classes, (1, 1), padding='same', name='conv2d_111')(x)
x = Activation('softmax', name='softmax')(x)
output = Reshape((num_classes,))(x)
model = Model(inputs=pre_model.input, outputs=output,  name='mobilenetv2_FER6')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

###################### loading fer2013 dataset ###############################################
# faces, emotions = load_fer2013()
# faces = preprocess_input(faces)
# x_train, x_test, y_train, y_test = train_test_split(faces, emotions,test_size=0.2, shuffle=False)
# x_PublicTest, x_PrivateTest, y_PublicTest, y_PrivateTest = train_test_split(x_test, y_test,test_size=0.5,shuffle=False)

###################### loading retrain dataset ###############################################
faces, emotions = load_retrain_data(retrain_data_path,
                                    file_ext='*.png',
                                    image_size=(input_shape[1], input_shape[0]), channel=input_shape[2],
                                    crop_face=True
                                    # crop_face=False,
                                    )
x_train, x_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.1, shuffle=True)

################callbacks##########################################################
# log_file_path = trained_models_path +'-'+'training_again.log'
# log_file_path = base_path + '_emotion_training.log'
logs = TensorBoard(log_dir=TensorBoard_logdir_path)   # 保存模型训练日志
# csv_logger = CSVLogger(log_file_path, append=False)
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience/3), verbose=1)
# model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
# model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1, save_best_only=True)

callbacks = [early_stop, reduce_lr, logs]

########################### Trainning ###########################################################################
History = model.fit_generator(data_generator.flow(x_train, y_train,
                                            batch_size),
                        steps_per_epoch=len(x_train) / batch_size,
                        epochs=num_epochs, verbose=1, callbacks=callbacks
                        ,validation_data=(x_test, y_test)
                    )

##############################画出loss曲线################################
# plt.plot(History.history['loss'],color='g',label='train_loss')
# plt.plot(History.history['val_loss'],color='b',label='val_loss')
# plt.legend(loc="lower right")
# plt.show()

############################## 输出训练好的模型在测试集上的表现#############
Test_score = model.evaluate(x_test, y_test)
#
print('PublicTest score:', Test_score[0])
print('PublicTest accuracy:', Test_score[1])

# PrivateTest_score = model.evaluate(x_PrivateTest, y_PrivateTest)
# print('PrivateTest score:', PrivateTest_score[0])
# print('PrivateTest accuracy:', PrivateTest_score[1])

######################## 保存模型 ##############################################
# Model_names = trained_models_path + '-' + '{0:.2f}'.format(PublicTest_score[1]) + '-' + \
#               '{0:.2f}'.format(PrivateTest_score[1])+'.hdf5'

Model_names = trained_models_path + '-' + '{0:.3f}'.format(Test_score[1])+'.hdf5'
model.save(Model_names)

######################## 画出测试集混淆矩阵 ##############################################
pred = model.predict(x_test)

# PrivateTest_score = model.evaluate(x_test, y_test)
# print('PublicTest accuracy:', PrivateTest_score[1])

# classes = ["angry" ,"disgust","fear", "happy", "sad", "surprised"]
# labels_name = ["Ang.", "Dis.", "Fear", "Hap.", "Sad", "Sur.", "Neu."]
labels_name = ["Ang.", "Dis.", "Fear", "Hap.", "Sad", "Sur."]

paintConfusion_float(ture_labels=y_test, pred_labels=pred,
                     labels_name=labels_name,
                     # save_path='./models/pictures/fer2013_PrivateTest1_{0:.3f}'.format(PrivateTest_score[1])+'.png'
                     )
