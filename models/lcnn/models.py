import tensorflow as tf
from keras.layers import Activation, Dense, BatchNormalization, MaxPool2D, Lambda, Input, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras.initializers import he_normal

from layers import Maxout


#function that return the stuck of Conv2D and MFM
def MaxOutConv2D(input, dim, kernel_size, strides, padding='same'):
    """MaxOutConv2D

    This is a helper function for LCNN class.
    This function combine Conv2D layer and Mac Feature Mapping function (MFM).
    Makes codes more readable.

    Args:
      input(tf.Tensor): The tensor from a previous layer.
      dim(int): Dimenstion of the Convolutional layer.
      kernel_size(int): Kernel size of Convolutional layer.
      strides(int): Strides for Convolutional layer.
      padding(string): Padding for Convolutional layer, "same" or "valid".

     Returns:
      mfm_out: Outputs after MFM.
       
    Examples:
      conv2d_1 = MaxOutConv2D(input, 64, kernel_size=2, strides=2, padding="same")

    """
    conv_out = Conv2D(dim, kernel_size=kernel_size, strides=strides, padding=padding)(input)
    mfm_out = Maxout(int(dim/2))(conv_out)
    return mfm_out


#function that return the stuck of FC and MFM
def MaxOutDense(x, dim):
    """ MaxOutDense
    
    Almost same as MaxOutConv2D.
    the difference is just Dense layer but not convolutional layer.

    """
    dense_out = Dense(dim)(x)
    mfm_out = Maxout(int(dim/2))(dense_out)
    return mfm_out

class LCNN(tf.keras.models.Model):
  def __init__(self):
    super(LCNN, self).__init__()
    self.conv2d_1 = Conv2D(64, kernel_size=5, strides=1, padding='same')
    self.mfm_1 = Maxout(32)
    self.maxpool_1 = MaxPool2D(pool_size=(2, 2), strides=(2,2))

    self.conv2d_2 = Conv2D(64, kernel_size=1, strides=1, padding='same')
    self.mfm_2 = Maxout(32)
    self.batch_norm_2 = BatchNormalization() 

    self.conv2d_3 = Conv2D(96, kernel_size=3, strides=1, padding='same')
    self.mfm_3 = Maxout(48)
    self.maxpool_3 = MaxPool2D(pool_size=(2, 2), strides=(2,2))
    self.batch_norm_3 = BatchNormalization() 

    self.conv2d_4 = Conv2D(96, kernel_size=1, strides=1, padding='same')
    self.mfm_4 = Maxout(48)
    self.batch_norm_4 = BatchNormalization()

    self.conv2d_5 = Conv2D(128, kernel_size=3, strides=1, padding='same')
    self.mfm_5 = Maxout(64)
    self.maxpool_5 = MaxPool2D(pool_size=(2, 2), strides=(2,2))

    self.conv2d_6 = Conv2D(128, kernel_size=1, strides=1, padding='same')
    self.mfm_6 = Maxout(64)
    self.batch_norm_6 = BatchNormalization()

    self.conv2d_7 = Conv2D(64, kernel_size=3, strides=1, padding='same')
    self.mfm_7 = Maxout(32)
    self.batch_norm_7 = BatchNormalization()

    self.conv2d_8 = Conv2D(64, kernel_size=1, strides=1, padding='same')
    self.mfm_8 = Maxout(32)
    self.batch_norm_8 = BatchNormalization()

    self.conv2d_9 = Conv2D(64, kernel_size=3, strides=1, padding='same')
    self.mfm_9 = Maxout(32)
    self.maxpool_9 = MaxPool2D(pool_size=(2, 2), strides=(2,2))
    self.flatten = Flatten()

    self.dense_10= Dense(160)
    self.mfm_10 = Maxout(80)
    self.batch_norm_10 = BatchNormalization()
    self.dropout_10 = Dropout(0.75)

    self.out = Dense(2, activation='softmax')


  def call(self, inputs, training=False):
    x = self.conv2d_1(inputs)
    x = self.mfm_1(x)
    x = self.maxpool_1(x)

    x = self.conv2d_2(inputs)
    x = self.mfm_2(x)
    x = self.batch_norm_2(x)

    x = self.conv2d_3(inputs)
    x = self.mfm_3(x)
    x = self.maxpool_3(x)
    x = self.batch_norm_3(x)

    x = self.conv2d_4(inputs)
    x = self.mfm_4(x)
    x = self.batch_norm_4(x)

    x = self.conv2d_5(inputs)
    x = self.mfm_5(x)
    x = self.maxpool_5(x)

    x = self.conv2d_6(inputs)
    x = self.mfm_6(x)
    x = self.batch_norm_6(x)

    x = self.conv2d_7(inputs)
    x = self.mfm_7(x)
    x = self.batch_norm_7(x)

    x = self.conv2d_8(inputs)
    x = self.mfm_8(x)
    x = self.batch_norm_8(x)

    x = self.conv2d_9(inputs)
    x = self.mfm_9(x)
    x = self.maxpool_9(x)
    
    x = self.flatten(x)

    x = self.dense_10(x)
    x = self.mfm_10(x)
    x = self.batch_norm_10(x)
    if training:
      x = self.dropout_10(x)

    return self.out(x)

# this function helps to build LCNN. 