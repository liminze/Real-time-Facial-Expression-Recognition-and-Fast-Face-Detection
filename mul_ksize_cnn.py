from keras.applications.imagenet_utils import _obtain_input_shape
from keras import backend as K
from keras.layers import Input, Convolution2D, GlobalAveragePooling2D, Dense, BatchNormalization, Activation
from keras.layers import DepthwiseConv2D,AveragePooling2D,MaxPool2D
from keras.models import Model
from keras.engine.topology import get_source_inputs
from keras.utils import plot_model
from keras.regularizers import l2
from keras.layers import Concatenate, Lambda
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout
from keras.layers import Activation, BatchNormalization, add, Reshape
from keras.applications.mobilenet import relu6, DepthwiseConv2D
from keras.utils.vis_utils import plot_model
from keras.layers import Lambda, Concatenate
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU, ThresholdedReLU


def channel_split(x):
    # equipartition
    in_channles = x.shape.as_list()[-1]
    ip = in_channles // 3
    c_s = Lambda(lambda z: z[:, :, :, 0:ip])(x)
    c_m = Lambda(lambda z: z[:, :, :, ip:2*ip])(x)
    c_l = Lambda(lambda z: z[:, :, :, 2*ip:])(x)
    return c_s, c_m ,c_l

def channel_shuffle(x):
    height, width, channels = x.shape.as_list()[1:]
    channels_per_split = channels // 3
    x = K.reshape(x, [-1, height, width, 3, channels_per_split])
    x = K.permute_dimensions(x, (0,1,2,4,3))
    x = K.reshape(x, [-1, height, width, channels])
    return x

def _conv_block(x, filters, kernel, strides):
    """Convolution Block
    This function defines a 2D convolution operation with BN and relu6.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
    # Returns
        Output tensor.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(filters, kernel, padding='same', strides=strides)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    # x = Activation(relu6)(x)
    x = PReLU()(x)
    # x = ELU()(x)
    return x


def _bottleneck(inputs, filters, kernel, t, s, r=False):
    """Bottleneck
    This function defines a basic bottleneck structure.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        r: Boolean, Whether to use the residuals.
    # Returns
        Output tensor.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    tchannel = K.int_shape(inputs)[channel_axis] * t

    x = _conv_block(inputs, tchannel, (1, 1), (1, 1))

    x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    # x = Activation(relu6)(x)
    x = PReLU()(x)
    # x = ELU()(x)

    x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)

    if r:
        x = add([x, inputs])
    return x


def _inverted_residual_block(inputs, filters, kernel, t, strides, n):
    """Inverted Residual Block
    This function defines a sequence of 1 or more identical layers.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        n: Integer, layer repeat times.
    # Returns
        Output tensor.
    """

    x = _bottleneck(inputs, filters, kernel, t, strides)

    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, 1, True)

    return x

def my_bottleneck(inputs, filters, kernel, strides=(2, 2)):

    x = Conv2D(filters, kernel_size=(1, 1), padding='same', strides=(1, 1))(inputs)
    x = BatchNormalization()(x)
    # x = Activation(relu6)(x)
    x = PReLU()(x)

    x = DepthwiseConv2D(kernel, strides=strides, depth_multiplier=1, padding='same')(x)
    x = BatchNormalization()(x)
    # x = Activation(relu6)(x)
    x = PReLU()(x)

    x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)

    return x

def mul_ksize_block(x, filters):
    # xs = my_bottleneck(x, filters, kernel=(5, 5))
    # xm = my_bottleneck(x, filters, kernel=(13, 13))
    # xl = my_bottleneck(x, filters, kernel=(21, 21))

    xs = _bottleneck(x, filters, kernel=(3, 3), t=3, s=2, r=False)
    xm = _bottleneck(x, filters, kernel=(7, 7), t=3, s=2, r=False)
    xl = _bottleneck(x, filters, kernel=(11, 11), t=3, s=2, r=False)
    x = Concatenate(axis=-1)([xs, xm, xl])
    return x

# def Multi_ksize_split(x):
#     xs,xm,xl = channel_split(x)
#     xs = _bottleneck(xs, int(xs.shape[-1]), kernel=(3, 3), t=3, s=2, r=False)
#     xm = _bottleneck(xm, int(xm.shape[-1]), kernel=(7, 7), t=3, s=2, r=False)
#     xl = _bottleneck(xl, int(xl.shape[-1]), kernel=(11, 11), t=3, s=2, r=False)
#     x = Concatenate(axis=-1)([xs, xm, xl])
#     # x = Lambda(channel_shuffle)(x)
#     return x

def Multi_ksize_split(x):
    xs = _bottleneck(x, int(x.shape[-1]), kernel=(3, 3), t=3, s=2, r=False)
    xm = _bottleneck(x, int(x.shape[-1]), kernel=(7, 7), t=3, s=2, r=False)
    xl = _bottleneck(x, int(x.shape[-1]), kernel=(11, 11), t=3, s=2, r=False)
    x = Concatenate(axis=-1)([xs, xm, xl])
    # x = Lambda(channel_shuffle)(x)
    return x

def mul3_ksize_block(input, filters):

    x = my_bottleneck(input, filters, kernel=(3, 3))
    x = _inverted_residual_block(x, 32, (3, 3), t=1, strides=1, n=2)
    x = _inverted_residual_block(x, 64, (3, 3), t=6, strides=2, n=3)
    xs = _inverted_residual_block(x, 96, (3, 3), t=6, strides=2, n=4)


    x = my_bottleneck(input, filters, kernel=(5, 5))
    x = _inverted_residual_block(x, 32, (5, 5), t=1, strides=1, n=2)
    x = _inverted_residual_block(x, 64, (5, 5), t=6, strides=2, n=3)
    xm = _inverted_residual_block(x, 96, (5, 5), t=6, strides=2, n=4)

    x = my_bottleneck(input, filters, kernel=(7, 7))
    x = _inverted_residual_block(x, 32, (7, 7), t=1, strides=1, n=2)
    x = _inverted_residual_block(x, 64, (7, 7), t=6, strides=2, n=3)
    xl = _inverted_residual_block(x, 96, (7, 7), t=6, strides=2, n=4)

    x = Concatenate(axis=-1)([xs, xm, xl])
    return x

def MUL_KSIZE_MobileNet_v2_best(input_shape, num_classes):

    """MobileNetv2
    xs = my_bottleneck(x, filters, kernel=(5, 5))
    xm = my_bottleneck(x, filters, kernel=(13, 13))
    xl = my_bottleneck(x, filters, kernel=(21, 21))

    # parameters
    batch_size = 16
    num_epochs = 500
    input_shape = (48, 48, 1)
    validation_split = .2
    verbose = 1
    num_classes = 7
    patience = 50

    # data generator
    data_generator = ImageDataGenerator(
                            featurewise_center=False,
                            featurewise_std_normalization=False,
                            rotation_range=10,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=.1,
                            horizontal_flip=True)
    """
    input = Input(shape=input_shape)
    x = mul_ksize_block(input, filters=48)
    x = _inverted_residual_block(x, 32, (3, 3), t=1, strides=1, n=1)
    x = _inverted_residual_block(x, 64, (3, 3), t=6, strides=2, n=3)
    x = _inverted_residual_block(x, 96, (3, 3), t=6, strides=2, n=4)
    x = _inverted_residual_block(x, 160, (3, 3), t=6, strides=2, n=3)
    x = _inverted_residual_block(x, 320, (3, 3), t=6, strides=1, n=1)
    x = _conv_block(x, 1280, (1, 1), strides=(1, 1))

    x = AveragePooling2D(pool_size=(3, 3))(x)
    # x = GlobalAveragePooling2D()(x)
    # x = Reshape((1, 1, int(x.shape[1])))(x)
    # x = Dropout(0.5, name='Dropout')(x)
    x = Conv2D(num_classes, (1, 1), padding='same')(x)
    x = Activation('softmax', name='softmax')(x)
    output = Reshape((num_classes,))(x)

    # x = GlobalAveragePooling2D()(x)
    # output = Dense(num_classes, activation='softmax')(x)

    model = Model(input, output)
    return model

def MUL_KSIZE_MobileNet_v2_mullayers(input_shape, num_classes):

    input = Input(shape=input_shape)
    x = mul_ksize_block(input, filters=16)
    x = _inverted_residual_block(x, 32, (3, 3), t=1, strides=1, n=1)
    x = _inverted_residual_block(x, 64, (3, 3), t=6, strides=2, n=3)
    x0 = _inverted_residual_block(x, 96, (3, 3), t=6, strides=2, n=4)
    x1 = _inverted_residual_block(x0, 160, (3, 3), t=6, strides=2, n=3)
    x2 = _inverted_residual_block(x1, 320, (3, 3), t=6, strides=1, n=1)
    x3 = _conv_block(x2, 1280, (1, 1), strides=(1, 1))

    n = int(x3.shape[1])
    X1 = Lambda(lambda X: tf.image.resize_images(X, size=(n, n)))(x1)
    X2 = Lambda(lambda X: tf.image.resize_images(X, size=(n, n)))(x2)
    # X4 = Lambda(lambda X: tf.image.resize_images(X, size=(n, n)))(x4)

    x = Concatenate(axis=-1)([X1, X2, x3])
    # x = Concatenate(axis=-1)([X1, x3, x4, x5])

    x = AveragePooling2D()(x)
    # x = GlobalAveragePooling2D()(x)
    # x = Reshape((1, 1, int(x.shape[1])))(x)
    # x = Dropout(0.3, name='Dropout')(x)
    x = Conv2D(num_classes, (1, 1), padding='same')(x)
    x = Activation('softmax', name='softmax')(x)
    output = Reshape((num_classes,))(x)

    # x = GlobalAveragePooling2D()(x)
    # output = Dense(num_classes, activation='softmax')(x)

    model = Model(input, output)
    return model

def MUL_KSIZE_MobileNet_v2_big(input_shape, num_classes):
    """MobileNetv2
    This function defines a MobileNetv2 architectures.
    # Arguments
        input_shape: An integer or tuple/list of 3 integers, shape
            of input tensor.
        num_classes: Integer, number of classes.
    # Returns
        MobileNetv2 model.
    """
    input = Input(shape=input_shape)
    x = mul3_ksize_block(input, filters=16)

    x = _inverted_residual_block(x, 160, (3, 3), t=6, strides=2, n=2)
    x = _inverted_residual_block(x, 320, (3, 3), t=6, strides=1, n=1)
    x = _conv_block(x, 1280, (1, 1), strides=(1, 1))

    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 1280))(x)
    x = Dropout(0.3, name='Dropout')(x)
    x = Conv2D(num_classes, (1, 1), padding='same')(x)

    x = Activation('softmax', name='softmax')(x)
    output = Reshape((num_classes,))(x)
    model = Model(input, output)
    return model

def MUL_KSIZE_MobileNet_v2(input_shape, num_classes):
    """MobileNetv2
    This function defines a MobileNetv2 architectures.
    # Arguments
        input_shape: An integer or tuple/list of 3 integers, shape
            of input tensor.
        num_classes: Integer, number of classes.
    # Returns
        MobileNetv2 model.
    """
    input = Input(shape=input_shape)
    x = _conv_block(input, 16, (3, 3), strides=(2, 2))
    x = _inverted_residual_block(x, 32, (3, 3), t=1, strides=1, n=1)
    x = _inverted_residual_block(x, 64, (3, 3), t=6, strides=2, n=3)
    x = _inverted_residual_block(x, 96, (3, 3), t=6, strides=1, n=4)
    x = Multi_ksize_split(x)
    x = _inverted_residual_block(x, 160, (3, 3), t=6, strides=2, n=3)
    x = _inverted_residual_block(x, 384, (3, 3), t=6, strides=1, n=1)
    x = Multi_ksize_split(x)



    x = _conv_block(x, 1280, (1, 1), strides=(1, 1))

    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 1280))(x)
    x = Dropout(0.5, name='Dropout')(x)
    x = Conv2D(num_classes, (1, 1), padding='same')(x)
    x = Activation('softmax', name='softmax')(x)
    output = Reshape((num_classes,))(x)

    # x = GlobalAveragePooling2D()(x)
    # output = Dense(num_classes, activation='softmax')(x)

    model = Model(input, output)
    return model

def MUL_KSIZE_shuffle_MobileNet_v2(input_shape, num_classes):
    input = Input(shape=input_shape)
    x = _conv_block(input, 48, (3, 3), strides=(1, 1))
    x = Multi_ksize_split(x)
    x = _inverted_residual_block(x, 32, (3, 3), t=3, strides=1, n=1)
    x = _inverted_residual_block(x, 96, (3, 3), t=3, strides=1, n=3)
    x = Multi_ksize_split(x)
    x = _inverted_residual_block(x, 160, (3, 3), t=3, strides=2, n=4)
    x = _inverted_residual_block(x, 288, (3, 3), t=3, strides=1, n=3)
    x = _inverted_residual_block(x, 432, (3, 3), t=3, strides=1, n=1)
    x = Multi_ksize_split(x)
    x = _conv_block(x, 1280, (1, 1), strides=(1, 1))
    # x = GlobalAveragePooling2D()(x)
    # output = Dense(num_classes, activation='softmax')(x)
#################################################################

    # input = Input(shape=input_shape)
    # x = _conv_block(input, 48, (3, 3), strides=(1, 1))
    # x = Multi_ksize_split(x)
    # x = _inverted_residual_block(x, 32, (3, 3), t=3, strides=1, n=1)
    # x = _inverted_residual_block(x, 96, (3, 3), t=3, strides=2, n=3)
    # x = Multi_ksize_split(x)
    # x = _inverted_residual_block(x, 160, (3, 3), t=3, strides=1, n=1)
    # x = _inverted_residual_block(x, 288, (3, 3), t=3, strides=2, n=3)
    # x = Multi_ksize_split(x)
    # x = _inverted_residual_block(x, 432, (3, 3), t=3, strides=1, n=1)
    # x = _conv_block(x, 1280, (1, 1), strides=(1, 1))

    # x = GlobalAveragePooling2D()(x)
    # x = Reshape((1, 1, 1280))(x)
    # x = Dropout(0.5, name='Dropout')(x)
    # x = Conv2D(num_classes, (1, 1), padding='same')(x)
    # x = Activation('softmax', name='softmax')(x)
    # output = Reshape((num_classes,))(x)

    x = GlobalAveragePooling2D()(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(input, output)
    return model


if __name__ == '__main__':
    input_shape = (48, 48, 1)
    num_classes = 7
    model = MUL_KSIZE_MobileNet_v2_best(input_shape, num_classes)
    # plot_model(model, 'models/pictures/MUL_KSIZE_MobileNet_v2_best.png', show_shapes=True, show_layer_names=True)  # 保存模型图
    model.summary()