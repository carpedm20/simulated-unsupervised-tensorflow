import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
import tf.nn.sigmoid_cross_entropy_with_logits as CE_loss

from layers import *

def to_float(layer):
  return tf.image.convert_image_dtype(layer, tf.float32)

class Network(object):
  def __init__(self, config):
    self.lambda_ = config.lambda_

    input_dims = [
        None, config.input_height,
        config.input_width, config.input_channel,
    ]

    self.x = to_float(tf.placeholder(tf.uint8, input_dims, 'x'))
    self.targets = to_float(tf.placeholder(tf.uint8, input_dims, 'targets'))

    self.layer_dict = {}

    with arg_scope([resnet_block, conv2d, max_pool2d], layer_dict=self.layer_dict):
      self.R_x = self._build_refiner_network()

      self.D_x = self._build_discriminator_network(self.x)
      self.D_R_x = self._build_discriminator_network(self.R_x)

      #self.estimate_outputs = self._build_estimation_network()

  def _build_loss(self):
    # Refiner loss
    real_label = tf.ones_like(self.D_R_x)
    fake_label = tf.zeros_like(self.D_R_x)

    with tf.variable_scope("refiner"):
      realism_loss = CE_loss(self.D_R_x, real_label)
      regularization_loss = \
          self.lambda_ * tf.reduce_sum(self.R_x - self.x, [1, 2, 3])

    # Discriminator loss
    with tf.variable_scope("discriminator"):
      refiner_d_loss = CE_loss(self.D_R_x, fake_label)
      synthetic_d_loss = CE_loss(self.D_x, real_label)

      self.discrim_loss = tf.reduce_mean(
          self.refiner_d_loss + self.synthetic_d_loss)

  def _build_refiner_network(self):
    layer = self.x
    with tf.variable_scope("refiner") as sc:
      layer = repeat(layer, 5, resnet_block, scope="resnet")
      layer = conv2d(layer, 1, 1, 1, scope="conv_1")
      self.refiner_variables = sc.get_variable()
    return layer

  def _build_discriminator_network(self, layer):
    with tf.variable_scope("discriminator") as sc:
      layer = conv2d(layer, 96, 3, 2, scope="conv_1")
      layer = conv2d(layer, 64, 3, 2, scope="conv_2")
      layer = max_pool2d(layer, 3, 1, scope="max_1")
      layer = conv2d(layer, 32, 3, 1, scope="conv_3")
      layer = conv2d(layer, 32, 1, 1, scope="conv_4")
      layer = conv2d(layer, 2, 1, 1, 
          activation_fn=tf.nn.softmax, scope="conv_5")
      self.discriminator_variables = sc.get_variable()
    return layer

  def _build_estimation_network(self):
    layer = self.x
    with tf.variable_scope("estimation"):
      layer = conv2d(layer, 96, 3, 2, scope="conv_1")
      layer = conv2d(layer, 64, 3, 2, scope="conv_2")
      layer = max_pool2d(layer, 64, 3, scope="max_1")
      layer = conv2d(layer, 32, 3, 1, scope="conv_3")
      layer = conv2d(layer, 32, 1, 1, scope="conv_4")
      layer = conv2d(layer, 2, 1, 1, activation_fn=slim.softmax)
    return layer
