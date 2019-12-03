import numpy as np
import os
from PIL import Image
from keras.models import Model
from nn.PetroNetModel import get_model

# extract the feature of the entire image
# input:    im, image to be segmented. (Note: im is a multi-angle image,
#               each angle is a single image in RGB color space)
# output:   fm, the feature maps of the input images
def nn_feature(im):
    # load the pre-trained model and network parameters
    model = get_model()
    model.load_weights('./model.h5')
    # we need the outputs after the last convolution layer (actually, the 2D pooling layer)
    model1 = Model(input=model.input, output=model.get_layer('out2').output)

    fm = []

    number_of_angle = len(im)

    for i in range(number_of_angle):

        gray_img = im.convert("L")
        arr = np.asarray(gray_img, dtype="float32")
        (h, w) = arr.shape
        data = np.empty((1, h, w, 1), dtype="float32")
        data[0, :, :, 0] = arr

        fm = model1.predict(data, verbose = 0)
    return fm
