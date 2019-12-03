import numpy as np
import tensorflow as tf
from keras.engine.topology import Layer
import keras.backend as K


class RoiMaxPooling(Layer):
    '''ROI pooling layer for 2D inputs.
    Transform the input image with arbitary size (w, h) into fixed-size output (W, H)
    
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 5 will result in a 5x5 region.
    # Input shape
        3D tensor X_img with shape:
        `(channels, rows, cols)` if dim_ordering='th'
        or 3D tensor with shape:
        `(rows, cols, channels)` if dim_ordering='tf'.
    # Output shape
        3D tensor with shape:
        `(channels, pool_size, pool_size)`
    '''

    def __init__(self, pool_size, **kwargs):
        self.dim_ordering = K.image_dim_ordering()
        assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.pool_size = pool_size
        super(RoiMaxPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'th':
            self.nb_channels = input_shape[1]
        elif self.dim_ordering == 'tf':
            self.nb_channels = input_shape[3]

    def compute_output_shape(self, input_shape):
        if self.dim_ordering == 'th':
            return None, self.nb_channels, self.pool_size, self.pool_size
        else:
            return None, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):
        input_shape = K.shape(x)
        if self.dim_ordering == 'th':
            w = input_shape[2]
            h = input_shape[3]
        elif self.dim_ordering == 'tf':
            w = input_shape[1]
            h = input_shape[2]
        w = K.cast(w, tf.float32)
        h = K.cast(h, tf.float32)

        # row_length = w / float(self.pool_size)
        # col_length = h / float(self.pool_size)
        # num_pool_regions = self.pool_size

        if self.dim_ordering == 'tf':
            rs = tf.image.resize_images(x, (self.pool_size, self.pool_size))
        else:
            print ("This version supports Tensorflow only.")

        return rs


    def get_config(self):
        config = {'pool_size': self.pool_size}
        base_config = super(RoiMaxPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
