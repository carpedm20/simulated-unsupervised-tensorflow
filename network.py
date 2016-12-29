import tensorflow as tf
from tensorflow.contrib.framework import arg_scope

from layers import *

class Network(object):
  def __init__(self, config):
    input_dims = [
        None, config.input_height,
        config.input_width, config.input_channel,
    ]

    def to_float(layer):
      return tf.image.convert_image_dtype(layer, tf.float32)

    self.inputs = to_float(tf.placeholder(tf.uint8, input_dims, 'inputs'))
    #self.input_real = to_float(tf.placeholder(tf.uint8, input_dims, 'input_real'))
    #self.input_synthetic = to_float(tf.placeholder(tf.uint8, input_dims, 'input_synthetic'))
    self.targets = to_float(tf.placeholder(tf.uint8, input_dims, 'targets'))

    self.layer_dict = {}

    with arg_scope([resnet_block, conv2d, max_pool2d], layer_dict=self.layer_dict):
      self.refiner_outputs = self._build_refiner_network()

      self.discrim_inputs = self._build_discriminator_network(self.inputs)
      self.discrim_refiner = self._build_discriminator_network(self.refiner_outputs)
      import ipdb; ipdb.set_trace() 

      #self.estimate_outputs = self._build_estimation_network()

    self.refiner_loss = tf.reduce_sum(self.refiner_outputs - self.inputs, [1, 2, 3])

  def _build_refiner_network(self):
    layer = self.inputs
    with tf.variable_scope("refiner"):
      layer = repeat(layer, 5, resnet_block, scope="resnet")
      layer = conv2d(layer, 1, 1, 1, scope="conv_1")
    return layer

  def _build_discriminator_network(self, layer):
    with tf.variable_scope("discriminator"):
      layer = conv2d(layer, 96, 3, 2, scope="conv_1")
      layer = conv2d(layer, 64, 3, 2, scope="conv_2")
      layer = max_pool2d(layer, 3, 1, scope="max_1")
      layer = conv2d(layer, 32, 3, 1, scope="conv_3")
      layer = conv2d(layer, 32, 1, 1, scope="conv_4")
      layer = conv2d(layer, 2, 1, 1, activation_fn=tf.nn.softmax, scope="conv_5")
    return layer

  def _build_estimation_network(self):
    layer = self.inputs
    with tf.variable_scope("estimation"):
      layer = conv2d(layer, 96, 3, 2, scope="conv_1")
      layer = conv2d(layer, 64, 3, 2, scope="conv_2")
      layer = max_pool2d(layer, 64, 3, scope="max_1")
      layer = conv2d(layer, 32, 3, 1, scope="conv_3")
      layer = conv2d(layer, 32, 1, 1, scope="conv_4")
      layer = conv2d(layer, 2, 1, 1, activation_fn=slim.softmax)
    return layer
