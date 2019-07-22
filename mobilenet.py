from keras.applications.imagenet_utils import _obtain_input_shape
from keras import backend as K
from keras.layers import Input, Convolution2D, GlobalAveragePooling2D, Dense, BatchNormalization, Activation
from keras.layers import DepthwiseConv2D,AveragePooling2D
from keras.models import Model
from keras.engine.topology import get_source_inputs
from keras.utils import plot_model
from keras.regularizers import l2
from keras.layers import Concatenate, Lambda
import tensorflow as tf

'''Google MobileNet model for Keras.
# Reference:
- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications]
(https://arxiv.org/pdf/1704.04861.pdf)
'''

# def MobileNet(input_shape=None, alpha=1, shallow=False, classes=1000):
def MobileNet(input_shape = None, num_classes = 7):
    """Instantiates the MobileNet.Network has two hyper-parameters
        which are the width of network (controlled by alpha)
        and input size.
        
        # Arguments
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            input_shape: optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape
                has to be `(224, 224, 3)` (with `channels_last` data format)
                or `(3, 224, 244)` (with `channels_first` data format).
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 96.
                E.g. `(200, 200, 3)` would be one valid value.
            alpha: optional parameter of the network to change the 
                width of model.
            shallow: optional parameter for making network smaller.
            classes: optional number of classes to classify images
                into.
        # Returns
            A Keras model instance.

        """

    # input_shape = _obtain_input_shape(input_shape,
    #                                   default_size=224,
    #                                   min_size=96,
    #                                   data_format=K.image_data_format(),
    #                                   # include_top=True)
    #                                   require_flatten=False)
    # if input_tensor is None:
    #     img_input = Input(shape=input_shape)
    # else:
    #     if not K.is_keras_tensor(input_tensor):
    #         img_input = Input(tensor=input_tensor, shape=input_shape)
    #     else:
    #         img_input = input_tensor
    alpha = 1
    # regularization = l2(0.01)

    img_input = Input(input_shape)
    x = Convolution2D(int(32 * alpha), (3, 3), strides=(2, 2), padding='same',
                      # kernel_regularizer=regularization,
                      use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same',
                        # kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(64 * alpha), (1, 1), strides=(1, 1), padding='same',
                      # kernel_regularizer=regularization,
                      use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv2D((3, 3), strides=(2, 2), padding='same',
                        # kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(128 * alpha), (1, 1), strides=(1, 1), padding='same',
                      # kernel_regularizer=regularization,
                      use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same',
                        # kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(128 * alpha), (1, 1), strides=(1, 1), padding='same',
                      # kernel_regularizer=regularization,
                      use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv2D((3, 3), strides=(2, 2), padding='same',
                        # kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(256 * alpha), (1, 1), strides=(1, 1), padding='same',
                      # kernel_regularizer=regularization,
                      use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same',
                        # kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(256 * alpha), (1, 1), strides=(1, 1), padding='same',
                      # kernel_regularizer=regularization,
                      use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv2D((3, 3), strides=(2, 2), padding='same',
                        # kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(512 * alpha), (1, 1), strides=(1, 1), padding='same',
                      # kernel_regularizer=regularization,
                      use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # if not shallow:
    #     for _ in range(5):
    #         x = DepthwiseConvolution2D(int(512 * alpha), (3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
    #         x = BatchNormalization()(x)
    #         x = Activation('relu')(x)
    #         x = Convolution2D(int(512 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    #         x = BatchNormalization()(x)
    #         x = Activation('relu')(x)

    x = DepthwiseConv2D((3, 3), strides=(2, 2), padding='same',
                        # kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(1024 * alpha), (1, 1), strides=(1, 1), padding='same',
                      # kernel_regularizer=regularization,
                      use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same',
                        # kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(1024 * alpha), (1, 1), strides=(1, 1), padding='same',
                      # kernel_regularizer=regularization,
                      use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # x = Convolution2D(128, (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    # x = Convolution2D(256, (3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
    #
    # x = Convolution2D(128, (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    # x = Convolution2D(256, (3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
    #
    # x = Convolution2D(128, (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    # x = Convolution2D(256, (3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
    #
    # x = Convolution2D(128, (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    # x = Convolution2D(256, (3, 3), strides=(2, 2), padding='same', use_bias=False)(x)

    # x = Convolution2D(classes, (3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
    x = GlobalAveragePooling2D()(x)
    output = Dense(num_classes, activation='softmax')(x)
    # output = Activation('softmax', name='predictions')(x)
    model = Model(img_input, output, name='mobilenet')

    return model


def MobileNet_MIX(input_shape=None, num_classes=7):
    alpha = 1
    # regularization = l2(0.01)

    img_input = Input(input_shape)
    x = Convolution2D(int(32 * alpha), (3, 3), strides=(2, 2), padding='same',
                      # kernel_regularizer=regularization,
                      use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x1 = Activation('relu')(x)

    x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same',
                        # kernel_regularizer=regularization,
                        use_bias=False)(x1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(64 * alpha), (1, 1), strides=(1, 1), padding='same',
                      # kernel_regularizer=regularization,
                      use_bias=False)(x)
    x = BatchNormalization()(x)
    x2 = Activation('relu')(x)

    x = DepthwiseConv2D((3, 3), strides=(2, 2), padding='same',
                        # kernel_regularizer=regularization,
                        use_bias=False)(x2)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(128 * alpha), (1, 1), strides=(1, 1), padding='same',
                      # kernel_regularizer=regularization,
                      use_bias=False)(x)
    x = BatchNormalization()(x)
    x3 = Activation('relu')(x)

    x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same',
                        # kernel_regularizer=regularization,
                        use_bias=False)(x3)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(128 * alpha), (1, 1), strides=(1, 1), padding='same',
                      # kernel_regularizer=regularization,
                      use_bias=False)(x)
    x = BatchNormalization()(x)
    x4 = Activation('relu')(x)

    x = DepthwiseConv2D((3, 3), strides=(2, 2), padding='same',
                        # kernel_regularizer=regularization,
                        use_bias=False)(x4)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(256 * alpha), (1, 1), strides=(1, 1), padding='same',
                      # kernel_regularizer=regularization,
                      use_bias=False)(x)
    x = BatchNormalization()(x)
    x5 = Activation('relu')(x)

    x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same',
                        # kernel_regularizer=regularization,
                        use_bias=False)(x5)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(256 * alpha), (1, 1), strides=(1, 1), padding='same',
                      # kernel_regularizer=regularization,
                      use_bias=False)(x)
    x = BatchNormalization()(x)
    x6 = Activation('relu')(x)

    x = DepthwiseConv2D((3, 3), strides=(2, 2), padding='same',
                        # kernel_regularizer=regularization,
                        use_bias=False)(x6)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(512 * alpha), (1, 1), strides=(1, 1), padding='same',
                      # kernel_regularizer=regularization,
                      use_bias=False)(x)
    x = BatchNormalization()(x)
    x7 = Activation('relu')(x)

    x = DepthwiseConv2D((3, 3), strides=(2, 2), padding='same',
                        # kernel_regularizer=regularization,
                        use_bias=False)(x7)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(1024 * alpha), (1, 1), strides=(1, 1), padding='same',
                      # kernel_regularizer=regularization,
                      use_bias=False)(x)
    x = BatchNormalization()(x)
    x8 = Activation('relu')(x)

    x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same',
                        # kernel_regularizer=regularization,
                        use_bias=False)(x8)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(1024 * alpha), (1, 1), strides=(1, 1), padding='same',
                      # kernel_regularizer=regularization,
                      use_bias=False)(x)
    x = BatchNormalization()(x)
    x9 = Activation('relu')(x)

    n = int(x9.shape[1])
    X1 = Lambda(lambda X: tf.image.resize_images(X, size=(n, n)))(x1)
    # X2 = Lambda(lambda X: tf.image.resize_images(X, size=(n, n)))(x2)
    X4 = Lambda(lambda X: tf.image.resize_images(X, size=(n, n)))(x4)
    # X6 = Lambda(lambda X: tf.image.resize_images(X, size=(n, n)))(x6)
    X7 = Lambda(lambda X: tf.image.resize_images(X, size=(n, n)))(x7)

    x = Concatenate(axis=-1)([X1, X4, X7, x9])

    # x = Convolution2D(classes, (3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
    x = GlobalAveragePooling2D()(x)
    output = Dense(num_classes, activation='softmax')(x)
    # output = Activation('softmax', name='predictions')(x)
    model = Model(img_input, output, name='mobilenet')

    return model

if __name__ == '__main__':
    input_shape = (300, 300, 1)
    num_classes = 7
    model = MobileNet(input_shape, num_classes)
    plot_model(model, 'models/MobileNet_MIX.png',show_shapes=True,show_layer_names=True)  # 保存模型图
    model.summary()