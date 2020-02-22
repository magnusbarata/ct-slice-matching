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

MODELS = {'DeepRanking': deepRanking,
          'SiameseModel': siameseModel,
          'InceptionV3': inceptionV3,
          'NormalizedInceptionV3': normalizedInceptionV3}
