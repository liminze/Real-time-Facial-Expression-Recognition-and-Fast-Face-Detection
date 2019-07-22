# -*- coding: utf-8 -*-
"""
Description: Train emotion classification model
"""
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from load_and_process import *
# from models.cnn import *
from sklearn.model_selection import train_test_split
from mul_ksize_cnn import *
from mobilenet import *
from mobilenet_v2 import *
import resnet
from keras import backend as K
import os
from shufflenet import ShuffleNet
from shufflenetv2 import *
from keras import optimizers
import inception_v4

def use_gpu():
    """Configuration for GPU"""
    from keras.backend.tensorflow_backend import set_session
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)   # 使用第一台GPU
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9  # GPU使用率为50%
    config.gpu_options.allow_growth = True    # 允许容量增长
    set_session(tf.InteractiveSession(config=config))

use_gpu()
# parameters
batch_size = 64
num_epochs = 500
input_shape = (48, 48, 1)
# input_shape = (224, 224, 3)
# validation_split = .2
verbose = 1
num_classes = 7
patience = 30

trained_model_name = 'MUL_KSIZE_no_shuffle_MobileNet_v2'
TensorBoard_logdir_path = './models/log/MUL_KSIZE_no_shuffle_MobileNet_v2'
base_path = 'models/'
trained_models_path = base_path + trained_model_name

# data generator
data_generator = ImageDataGenerator(
                        featurewise_center=True,
                        # featurewise_std_normalization=True,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        # rescale=1./255,
                        zoom_range=.1,
                        horizontal_flip=True,
                        fill_mode='nearest')

# data_generator = ImageDataGenerator(
#                         featurewise_center=True,
#                         featurewise_std_normalization=False,
#                         rotation_range=30,
#                         width_shift_range=0.2,
#                         height_shift_range=0.2,
#                         shear_range=0.2,
#                         zoom_range=.2,
#                         horizontal_flip=True)

########### model parameters/compilation
model = MUL_KSIZE_MobileNet_v2_best(input_shape=input_shape, num_classes=num_classes)
# model = MUL_KSIZE_shuffle_MobileNet_v2(input_shape=input_shape, num_classes=num_classes)
# model = resnet.ResnetBuilder.build_resnet_101(input_shape=input_shape, num_outputs=num_classes)
# model = ShuffleNetV2(input_shape=input_shape,classes=num_classes)
# model = MobileNet_v2(input_shape=input_shape, num_classes=num_classes)
# model = Multi_ShuffleNetV2(input_shape=input_shape,classes=num_classes)
# model = inception_v4.create_model(input_shape=input_shape, num_classes=num_classes)

# sgd = optimizers.sgd(lr=0.001, momentum=0.9, nesterov=True)  # 优化器为SGD
# model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# callbacks
# log_file_path = trained_models_path +'-'+'training.log'
logs = TensorBoard(log_dir=TensorBoard_logdir_path)   # 保存模型训练日志
# csv_logger = CSVLogger(log_file_path, append=False)
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience/3), verbose=1)

model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'

model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1, save_best_only=True)
# callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr, logs]
# callbacks = [csv_logger, early_stop, reduce_lr, logs]
callbacks = [early_stop, reduce_lr, logs]


# loading dataset
faces, emotions = load_fer2013()
faces = preprocess_input(faces)
# faces = preprocess_input_0(faces)
# num_samples, num_classes = emotions.shape
x_train, x_test, y_train, y_test = train_test_split(faces, emotions,test_size=0.2, shuffle=False)
x_PublicTest, x_PrivateTest, y_PublicTest, y_PrivateTest = train_test_split(x_test, y_test,test_size=0.5,shuffle=False)

model.fit_generator(data_generator.flow(x_train, y_train,
                                            batch_size),
                        steps_per_epoch=len(x_train) / batch_size,
                        epochs=num_epochs, verbose=1, callbacks=callbacks,
                        validation_data=(x_PublicTest, y_PublicTest))


# # 训练模型
# his = model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs,
#                 verbose=1, callbacks=callbacks,
#                 # validation_data=(x_PublicTest,y_PublicTest))
#                 validation_data=(x_test,y_test))
# # print(his.history)


# PublicTest_score = model.evaluate(x_test, y_test)
# print('PublicTest score:', PublicTest_score[0])
# print('PublicTest accuracy:', PublicTest_score[1])
# Model_names = trained_models_path + '-' + '{0:.2f}'.format(PublicTest_score[1]) + '.hdf5'
# model.save(Model_names)


# 输出训练好的模型在测试集上的表现
PublicTest_score = model.evaluate(x_PublicTest, y_PublicTest)
print('PublicTest score:', PublicTest_score[0])
print('PublicTest accuracy:', PublicTest_score[1])

PrivateTest_score = model.evaluate(x_PrivateTest, y_PrivateTest)
print('PrivateTest score:', PrivateTest_score[0])
print('PrivateTest accuracy:', PrivateTest_score[1])

# 保存模型
Model_names = trained_models_path + '-' + '{0:.4f}'.format(PublicTest_score[1]) + '-' + \
              '{0:.4f}'.format(PrivateTest_score[1])+'.hdf5'
model.save(Model_names)
