import tensorflow as tf
from tensorflow.contrib.framework import arg_scope

from layers import *

class Model(object):
  def __init__(self, config):
    self.task = config.task
    self.debug = config.debug
    self.reg_scale = config.reg_scale
    self.learning_rate = config.learning_rate

    self.layer_dict = {}

    input_dims = [
        None, config.input_height,
        config.input_width, config.input_channel,
    ]

    self.x = tf.placeholder(tf.uint8, [None, None, None, config.input_channel], 'x')
    self.x_history = tf.placeholder(tf.uint8, [None, None, None, config.input_channel], 'x_history')

    resize_dim = [config.input_height, config.input_width]
    self.resized_x = tf.image.resize_images(self.x, resize_dim)
    self.resized_x_history = tf.image.resize_images(self.x_history, resize_dim)

    self.normalized_x = normalize(self.resized_x)
    self.normalized_x_history = normalize(self.resized_x_history)

    self._build_model()
    self._build_steps()

  def build_optim(self):
    self.refiner_step = tf.Variable(0, name='refiner_step', trainable=False)
    self.discrim_step = tf.Variable(0, name='discrim_step', trainable=False)

    if self.task == "generative":
      optim = tf.train.GradientDescentOptimizer(self.learning_rate)
      self.refiner_optim = optim.minimize(
          self.refiner_loss,
          global_step=self.refiner_step,
          var_list=self.refiner_vars,
      )

      optim = tf.train.GradientDescentOptimizer(self.learning_rate)
      self.discrim_optim = optim.minimize(
          self.discrim_loss,
          global_step=self.discrim_step,
          var_list=self.discrim_vars,
      )
    elif self.task == "estimate":
      raise Exception("[!] Not implemented yet")

  def _build_model(self):
    with arg_scope([resnet_block, conv2d, max_pool2d],
                   layer_dict=self.layer_dict):
      self.R_x = self._build_refiner()
      self.denormalized_R_x = denormalize(self.R_x)

      self.D_x, self.D_x_logits = \
          self._build_discrim(self.normalized_x, name="D_x")
      self.D_R_x, self.D_R_x_logits = \
          self._build_discrim(self.R_x, name="D_R_x", reuse=True)
      self.D_x_history, self.D_x_history_logits = \
          self._build_discrim(self.normalized_x_history, name="D_x_history", reuse=True)

      #self.estimate_outputs = self._build_estimation_network()
    self._build_loss()

  def _build_loss(self):
    # Refiner loss
    zeros = tf.zeros_like(self.D_R_x)[:,:,:,0]
    ones = tf.ones_like(self.D_R_x)[:,:,:,0]

    fake_label = tf.stack([zeros, ones], axis=-1)
    real_label = tf.stack([ones, zeros], axis=-1)

    with tf.variable_scope("refiner"):
      self.realism_loss = tf.reduce_sum(
          SE_loss(self.D_R_x_logits, real_label), [1, 2], name="realism_loss")
      self.regularization_loss = 0
      #    self.reg_scale * tf.reduce_sum(
      #        self.R_x - self.normalized_x, [1, 2, 3],
      #        name="regularization_loss")

      self.refiner_loss = tf.reduce_mean(
          self.realism_loss, #+ self.regularization_loss,
          name="refiner_loss")

      if self.debug:
        self.refiner_loss = tf.Print(self.refiner_loss, [self.R_x], "R_x")
        self.refiner_loss = tf.Print(self.refiner_loss, [self.D_R_x], "D_R_x")
        self.refiner_loss = tf.Print(self.refiner_loss, [self.normalized_x], "normalized_x")
        self.refiner_loss = tf.Print(self.refiner_loss, [self.regularization_loss], "reg_loss")

    self.refiner_summary = tf.summary.merge([
        tf.summary.image("input_images", self.x),
        tf.summary.image("refined_images", self.denormalized_R_x),
        tf.summary.scalar("realism_loss", tf.reduce_mean(self.realism_loss)),
        tf.summary.scalar("regularization_loss", tf.reduce_mean(self.regularization_loss)),
        tf.summary.scalar("loss", tf.reduce_mean(self.refiner_loss)),
    ])

    # Discriminator loss
    with tf.variable_scope("discriminator"):
      self.refiner_d_loss = tf.reduce_sum(
          SE_loss(self.D_R_x_logits, fake_label), [1, 2],
          name="refiner_d_loss")
      self.refiner_history_d_loss = tf.reduce_sum(
          SE_loss(self.D_x_history_logits, fake_label), [1, 2],
          name="refiner_history_d_loss")
      self.synthetic_d_loss = tf.reduce_sum(
          SE_loss(self.D_x_logits, real_label), [1, 2],
          name="synthetic_d_loss")

      self.discrim_loss = tf.reduce_mean(
          self.refiner_d_loss + \
              self.synthetic_d_loss, name="discrim_loss")

      self.discrim_loss_with_history = tf.reduce_mean(
          self.refiner_d_loss + self.refiner_history_d_loss + \
              self.synthetic_d_loss, name="discrim_loss_with_history")

      self.discrim_summary = tf.summary.merge([
          tf.summary.scalar("refiner_d_loss", self.refiner_d_loss),
          tf.summary.scalar("refiner_history_d_loss", self.refiner_history_d_loss),
          tf.summary.scalar("synthetic_d_loss", self.synthetic_d_loss),
          tf.summary.scalar("discrim_loss", self.discrim_loss),
          tf.summary.scalar("discrim_loss_with_history", self.discrim_loss_with_history),
      ])

  def _build_steps(self):
    def run(sess, inputs, fetch, input_op,
            summary_op, summary_writer, output_op=None):
      if summary_writer is not None:
        fetch['summary'] = summary_op
      if output_op is not None:
        fetch['output'] = output_op

      result = sess.run(fetch, feed_dict={ input_op: inputs })
      if result.has_key('summary'):
        summary_writer.add_summary(result['summary'], result['step'])
        summary_writer.flush()
      return result

    def train_refiner(sess, inputs, summary_writer=None, with_output=False):
      fetch = {
          'loss': self.refiner_loss,
          'optim': self.refiner_optim,
          'step': self.refiner_step,
      }
      return run(sess, inputs, fetch, self.x,
                 self.refiner_summary, summary_writer,
                 output_op=self.R_x if with_output else None)

    def test_refiner(sess, inputs, summary_writer=None, with_output=False):
      fetch = {
          'loss': self.refiner_loss,
          'step': self.refiner_step,
      }
      return run(sess, inputs, fetch, self.x,
                 self.refiner_summary, summary_writer,
                 output_op=self.R_x if with_output else None)

    def train_discrim(sess, inputs, summary_writer=None):
      fetch = {
          'loss': self.discrim_loss,
          'optim': self.discrim_optim,
          'step': self.discrim_step,
      }
      return run(sess, inputs, fetch, self.x,
                 self.discrim_summary, summary_writer)

    def test_discrim(sess, inputs, summary_writer=None):
      fetch = {
          'loss': self.discrim_loss,
          'step': self.discrim_step,
      }
      return run(sess, inputs, fetch, self.x,
                 self.discrim_summary, summary_writer=summary_writer)

    self.train_refiner = train_refiner
    self.test_refiner = test_refiner
    self.train_discrim = train_discrim
    self.test_discrim = test_discrim

  def _build_refiner(self):
    layer = self.normalized_x
    with tf.variable_scope("refiner") as sc:
      layer = repeat(layer, 5, resnet_block, scope="resnet")
      output = conv2d(layer, 1, 1, 1, scope="conv_1")
      self.refiner_vars = tf.contrib.framework.get_variables(sc)
    return output 

  def _build_discrim(self, layer, name, reuse=False):
    with tf.variable_scope("discriminator") as sc:
      with arg_scope([conv2d], weights_initializer=tf.contrib.layers.xavier_initializer()):
        layer = conv2d(layer, 96, 3, 2, scope="conv_1", name=name, reuse=reuse)
        layer = conv2d(layer, 64, 3, 2, scope="conv_2", name=name, reuse=reuse)
        layer = max_pool2d(layer, 3, 1, scope="max_1", name=name)
        layer = conv2d(layer, 32, 3, 1, scope="conv_3", name=name, reuse=reuse)
        layer = conv2d(layer, 32, 1, 1, scope="conv_4", name=name, reuse=reuse)
        logits = conv2d(layer, 2, 1, 1, scope="conv_5", name=name, reuse=reuse)
        output = tf.nn.softmax(logits, name="softmax")
        self.discrim_vars = tf.contrib.framework.get_variables(sc)
    return output, logits

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
