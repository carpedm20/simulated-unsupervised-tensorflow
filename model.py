import tensorflow as tf
from tensorflow.contrib.framework import arg_scope

from layers import *

class Model(object):
  def __init__(self, config):
    self.task = config.task
    self.reg_scale = config.reg_scale
    self.learning_rate = config.learning_rate

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
    self._build_steps()

  def build_optim(self):
    self.global_step = tf.Variable(0, name='global_step', trainable=False)

    if self.task == "generative":
      optim = tf.train.GradientDescentOptimizer(self.learning_rate)
      self.refiner_optim = optim.minimize(
          self.refiner_loss,
          global_step=self.global_step,
          var_list=self.refiner_vars,
      )

      optim = tf.train.GradientDescentOptimizer(self.learning_rate)
      self.discrim_optim = optim.minimize(
          self.discrim_loss,
          global_step=self.global_step,
          var_list=self.discrim_vars,
      )
    elif self.task == "estimate":
      raise Exception("[!] Not implemented yet")

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
      self.realism_loss = tf.reduce_sum(
          CE_loss(self.D_R_x, real_label), [1, 2, 3], name="realism_loss")
      self.regularization_loss = \
          self.reg_scale * tf.reduce_sum(
              self.R_x - self.normalized_x, [1, 2, 3],
              name="regularization_loss")

      self.refiner_loss = tf.reduce_mean(
          self.realism_loss - self.regularization_loss,
          name="refiner_loss")

      self.refiner_summary = tf.summary.merge([
          tf.summary.scalar("realism_loss", self.realism_loss),
          tf.summary.scalar("regularization_loss", self.regularization_loss),
          tf.summary.scalar("loss", self.refiner_loss),
      ])

    # Discriminator loss
    with tf.variable_scope("discriminator"):
      self.refiner_loss = tf.reduce_sum(
          CE_loss(self.D_R_x, fake_label), [1, 2, 3],
          name="refiner_loss")
      self.refiner_history_loss = tf.reduce_sum(
          CE_loss(self.D_x_history, fake_label), [1, 2, 3],
          name="refiner_history_loss")
      self.synthetic_loss = tf.reduce_sum(
          CE_loss(self.D_x, real_label), [1, 2, 3],
          name="synthetic_loss")

      self.discrim_loss = tf.reduce_mean(
          self.refiner_loss + self.synthetic_loss, name="discrim_loss")

      self.discrim_loss_with_history = tf.reduce_mean(
          self.refiner_loss + self.refiner_history_loss + \
              self.synthetic_loss, name="discrim_loss_with_history")

      self.discrim_summary = tf.summary.merge([
          tf.summary.scalar("refiner_loss", self.refiner_loss),
          tf.summary.scalar("refiner_history_loss", self.refiner_history_loss),
          tf.summary.scalar("synthetic_loss", self.synthetic_loss),
          tf.summary.scalar("discrim_loss", self.discrim_loss),
          tf.summary.scalar("discrim_loss_with_history", self.discrim_loss_with_history),
      ])

  def _build_steps(self):
    def run(sess, inputs, to_return, input_op, summary_op=None, output_op=None):
      if summary_op is not None:
        to_return += [summary_op]
      if output_op is not None:
        to_return += [output_op]
      return sess.run(to_return, feed_dict={ input_op: inputs })

    def train_refiner(sess, inputs, with_summary=False, with_output=False):
      return run(sess, inputs,
                 [self.refiner_loss, self.refiner_optim], self.x, 
                 summary_op=self.refiner_summary if with_summary else None,
                 output_op=self.R_x if with_output else None)

    def test_refiner(sess, inputs, with_summary=False, with_output=False):
      return run(sess, inputs,
                 [self.refiner_loss], self.x,
                 summary_op=self.refiner_summary if with_summary else None,
                 output_op=self.R_x if with_output else None)

    def train_discrim(sess, inputs, with_summary=False):
      return run(self, sess, inputs,
                 [self.discrim_loss, self.discrim_optim], self.x,
                 summary_op=self.discrim_summary if with_summary else None)

    def test_discrim(sess, inputs, with_summary=False):
      return run(sess, inputs,
                 [self.discrim_loss], self.x,
                 summary_op=self.discrim_summary if with_summary else None)

    self.train_refiner = train_refiner
    self.test_refiner = test_refiner
    self.train_discrim = train_discrim
    self.test_discrim = test_discrim

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
