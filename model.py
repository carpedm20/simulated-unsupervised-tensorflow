import tensorflow as tf
from tensorflow.contrib.framework import arg_scope

from layers import *

def normalize(layer):
  return tf.image.convert_image_dtype(layer, tf.float32)

class Model(object):
  def __init__(self, config):
    self.reg_scale = config.reg_scale

    self.layer_dict = {}

    input_dims = [
        None, config.input_height,
        config.input_width, config.input_channel,
    ]

    self.x = tf.placeholder(tf.uint8, input_dims, 'x')
    self.x_history = tf.placeholder(tf.uint8, input_dims, 'x_history')

    self.normalized_x = normalize(self.x)
    self.normalized_x_history = normalize(self.x_history)

    self._build_model()

    def run_step(self, sess, optim, to_return,
                 inputs, targets, input_op, target_op,
                 summary_op=None):
      if summary_op is not None:
        to_return += [summary_op]
      return sess.run(to_return,
                      feed_dict={input_op: inputs,
                                 target_op: targets})

    self._runs = {
      'generative_train': None,
      'generative_test': None,
      'estimate_train': None,
      'estimate_test': None,
    }

    self.train_step = self._runs['{}_train'.format(config.task_type)]
    self.test_step = self._runs['{}_test'.format(config.task_type)]

  def _build_model(self):
    with arg_scope([resnet_block, conv2d, max_pool2d],
                   layer_dict=self.layer_dict):
      self.R_x = self._build_refiner()

      self.D_x = self._build_discrim(self.normalized_x)
      self.D_R_x = self._build_discrim(self.R_x)
      self.D_x_history = self._build_discrim(self.normalized_x_history)

      #self.estimate_outputs = self._build_estimation_network()
    self._build_loss()

  def _build_loss(self):
    # Refiner loss
    real_label = tf.ones_like(self.D_R_x)
    fake_label = tf.zeros_like(self.D_R_x)

    with tf.variable_scope("refiner"):
      self.realism_r_loss = tf.reduce_sum(
          CE_loss(self.D_R_x, real_label), [1, 2, 3], name="realism_r_loss")
      self.regularization_r_loss = \
          self.reg_scale * tf.reduce_sum(
              self.R_x - self.normalized_x, [1, 2, 3],
              name="regularization_r_loss")

      self.refiner_loss = tf.reduce_mean(
          self.realism_r_loss - self.regularization_r_loss,
          name="refiner_loss")

    # Discriminator loss
    with tf.variable_scope("discriminator"):
      self.refiner_d_loss = tf.reduce_sum(
          CE_loss(self.D_R_x, fake_label), [1, 2, 3],
          name="refiner_d_loss")
      self.refiner_d_history_loss = tf.reduce_sum(
          CE_loss(self.D_x_history, fake_label), [1, 2, 3],
          name="refiner_d_history_loss")
      self.synthetic_d_loss = tf.reduce_sum(
          CE_loss(self.D_x, real_label), [1, 2, 3],
          name="synthetic_d_loss")

      self.discrim_loss = tf.reduce_mean(
          self.refiner_d_loss + self.synthetic_d_loss, name="discrim_loss")

      self.discrim_loss_with_history = tf.reduce_mean(
          self.refiner_d_loss + self.refiner_d_history_loss + \
              self.synthetic_d_loss, name="discrim_loss_with_history")

  def _build_refiner(self):
    layer = self.normalized_x
    with tf.variable_scope("refiner") as sc:
      layer = repeat(layer, 5, resnet_block, scope="resnet")
      layer = conv2d(layer, 1, 1, 1, scope="conv_1")
      self.refiner_vars = tf.contrib.framework.get_variables(sc)
    return layer

  def _build_discrim(self, layer):
    with tf.variable_scope("discriminator") as sc:
      layer = conv2d(layer, 96, 3, 2, scope="conv_1")
      layer = conv2d(layer, 64, 3, 2, scope="conv_2")
      layer = max_pool2d(layer, 3, 1, scope="max_1")
      layer = conv2d(layer, 32, 3, 1, scope="conv_3")
      layer = conv2d(layer, 32, 1, 1, scope="conv_4")
      layer = conv2d(layer, 2, 1, 1, 
          activation_fn=tf.nn.softmax, scope="conv_5")
      self.discrim_vars = tf.contrib.framework.get_variables(sc)
    return layer

  def _build_estimation_network(self):
    layer = self.normalized_x
    with tf.variable_scope("estimation"):
      layer = conv2d(layer, 96, 3, 2, scope="conv_1")
      layer = conv2d(layer, 64, 3, 2, scope="conv_2")
      layer = max_pool2d(layer, 64, 3, scope="max_1")
      layer = conv2d(layer, 32, 3, 1, scope="conv_3")
      layer = conv2d(layer, 32, 1, 1, scope="conv_4")
      layer = conv2d(layer, 2, 1, 1, activation_fn=slim.softmax)
    return layer
