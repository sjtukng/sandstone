from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers.core import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, Input, concatenate
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers.core import Lambda
from keras.optimizers import SGD
from keras.utils import plot_model

from nn.RoiMaxPooling import RoiMaxPooling


color_channel = 1
num_of_class = 3

#################################################################################################
# Create the base neural network which outputs the feature maps
#################################################################################################
def nn_base(input_tensor = None, trainable = False):

    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (color_channel, None, None)
    else:
        input_shape = (None, None, color_channel)

    if input_tensor is None:
        img_input = Input(shape = input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor = input_tensor, shape = input_shape)
        else:
            img_input = input_tensor

    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    x = Conv2D(8, (5, 5), activation='tanh', name='conv1')(img_input)
    x = Conv2D(16, (3, 3), activation='tanh', name='conv2')(x)
    out1 = MaxPooling2D((2, 2), name='pool1')(x)

    x = Conv2D(16, (3, 3), activation='tanh', name='conv3')(out1)
    out2 = MaxPooling2D((2, 2), name='pool2')(x)

    return [out1, out2]

#################################################################################################
# Create the classifer which take the feature maps as input and output the softmax result
#################################################################################################
def classifier(shared_features, nb_classes = 3, trainable = False, include_top = True):

    pooling_regions = 5

    out1 = RoiMaxPooling(pooling_regions, name='roi1')(shared_features[0])
    out2 = RoiMaxPooling(pooling_regions, name='roi2')(shared_features[1])

    out = concatenate([out1, out2])

    x = Flatten(name='flat')(out2)

    x = Dense(400, activation=('relu'), name='fc1')(x)
    x = Dense(20, activation=('relu'), name='fc2')(x)
    prediction = Dense(3, activation='softmax')(x)

    return prediction

#################################################################################################
# Create the model
#################################################################################################
def get_model():
    if K.image_dim_ordering() == 'th':
        input_shape_img = (color_channel, None, None)
    else:
        input_shape_img = (None, None, color_channel)

    img_input = Input(shape=input_shape_img)
    feature_maps = nn_base(img_input, trainable = True)
    clf = classifier(feature_maps, nb_classes = num_of_class, trainable = True)

    model = Model(inputs = img_input, outputs = clf)
    model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

# model = get_model()
# model.summary()
# plot_model(model, "PetroNet.png", show_shapes=True, show_layer_names=True)