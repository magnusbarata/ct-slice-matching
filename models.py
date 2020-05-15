import keras
from keras.layers import *
from keras.models import Model, load_model
import tensorflow as tf

def inceptionV3(shape):
    return keras.applications.inception_v3.InceptionV3(include_top=False, weights=None, input_shape=shape, pooling='avg')

def normalizedInceptionV3(shape):
    base_model = keras.applications.inception_v3.InceptionV3(include_top=False, weights=None, input_shape=shape, pooling='avg')
    l2_norm = Lambda(lambda  x: tf.nn.l2_normalize(x, axis=1))(base_model.output)
    return Model(base_model.input, l2_norm, name=base_model.name+'-l2normalized')

def deepRanking(shape, cnn_model=inceptionV3, emb_size=2048):
    """TODO: explanation, shift mean of l2 to 0"""
    input = Input(shape, dtype='float')
    cnn = cnn_model(shape)
    cnn_out = cnn(input)
    cnn_l2 = Lambda(lambda  x: tf.nn.l2_normalize(x, axis=1))(cnn_out)

    x1 = AveragePooling2D((4,4))(input)
    x1 = Conv2D(96, (8,8), strides=(4,4))(x1)
    x1 = MaxPool2D((4,4))(x1)

    x2 = AveragePooling2D((8,8))(input)
    x2 = Conv2D(96, (8,8), strides=(4,4))(x2)
    x2 = MaxPool2D((2,2))(x2)
    x12_l2 = Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(Concatenate()([Flatten()(x1), Flatten()(x2)]))

    dense = Dense(emb_size)(Concatenate()([cnn_l2, x12_l2]))
    final_l2 = Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(dense)
    return Model(input, final_l2, name=cnn.name+'-DeepRanking')

def _l1distance(var):
    return tf.abs(var[0]-var[1])

def _l2distance(var):
    return (var[0]-var[1]) ** 2

def siameseModel(shape, base_model=deepRanking, distance_function=_l2distance):
    in1 = Input(shape, dtype='float')
    in2 = Input(shape, dtype='float')

    base = base_model(shape)
    out1 = base(in1)
    out2 = base(in2)

    dist = Lambda(distance_function)([out1, out2])
    prediction = Dense(1, activation='sigmoid')(dist)
    return Model([in1, in2], prediction, name=base.name+'-Siamese')

def learnMetric(shape):
    def shared_net1(shape):
        input = Input(shape, dtype='float')
        conv1 = SeparableConv2D(32, (5,5))(input)
        conv1 = MaxPool2D((3,3))(conv1)
        conv1 = Conv2D(32, (1,1))(conv1)
        conv2 = SeparableConv2D(64, (3,3))(conv1)
        conv2 = MaxPool2D((2,2))(conv2)
        output = Conv2D(64, (1,1))(conv2)

        return Model(input, output, name='shared_net1')

    def shared_net2(shape):
        input =Input(shape, dtype='float')
        conv3 = SeparableConv2D(32, (3,3))(input)
        conv3 = MaxPool2D((2,2))(conv3)
        conv3 = Conv2D(32, (1,1))(conv3)
        conv4 = SeparableConv2D(32, (3,3))(conv3)
        conv4 = MaxPool2D((2,2))(conv4)
        conv4 = Conv2D(32, (1,1))(conv4)
        conv5 = SeparableConv2D(32, (3,3))(conv4)
        conv5 = MaxPool2D((2,2))(conv5)
        conv5 = Conv2D(16, (1,1))(conv5)
        fc = Flatten()(conv5)
        fc = Dense(500)(fc)
        fc = Dense(500)(fc)
        output = Dense(1, activation='sigmoid')(fc)

        return Model(input, output, name='shared_net2')

    in_r = Input(shape, dtype='float')
    in_n = Input(shape, dtype='float')
    in_p = Input(shape, dtype='float')

    # Encode positive, reference, and negative
    shared1 = shared_net1(shape)
    encoded_r = shared1(in_r)
    encoded_n = shared1(in_n)
    encoded_p = shared1(in_p)

    # Regroup pairs
    neg_pair = Concatenate()([encoded_r, encoded_n])
    pos_pair = Concatenate()([encoded_r, encoded_p])

    # Encode positive and negative pair
    shared2 = shared_net2(keras.backend.int_shape(neg_pair)[1:])
    encoded_neg = shared2(neg_pair)
    encoded_pos = shared2(pos_pair)
    out = Concatenate()([encoded_neg, encoded_pos])

    return Model([in_r, in_n, in_p], out, name='LearnMetric')

MODELS = {'DeepRanking': deepRanking,
          'SiameseModel': siameseModel,
          'InceptionV3': inceptionV3,
          'NormalizedInceptionV3': normalizedInceptionV3,
          'LearnMetric': learnMetric}
