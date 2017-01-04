import tensorflow as tf
from tensorflow.contrib.framework import arg_scope

from layers import *
from utils import show_all_variables

class Model(object):
  def __init__(self, config, data_loader):
    self.task = config.task
    self.debug = config.debug
    self.reg_scale = config.reg_scale
    self.learning_rate = config.learning_rate
    self.max_grad_norm = config.max_grad_norm

    self.layer_dict = {}

    image_dims = [config.input_height, config.input_width, config.input_channel]

    min_after_dequeue = 5000
    capacity = min_after_dequeue + 3 * config.batch_size

    self.synthetic_batch_size = tf.placeholder(tf.int32, [], "synthetic_batch_size")
    self.synthetic_images = image_from_paths(data_loader.synthetic_data_paths,
                                             data_loader.synthetic_data_dims)
    self.x = tf.train.shuffle_batch(
        [self.synthetic_images], batch_size=self.synthetic_batch_size,
        num_threads=4, capacity=capacity,
        min_after_dequeue=min_after_dequeue, name='synthetic_inputs')

    self.y = tf.placeholder(
        tf.uint8, [None, None, None, config.input_channel], name='real_inputs')
    self.R_x_history = tf.placeholder(
        tf.float32, [None, None, None, config.input_channel], 'R_x_history')

    resize_dim = [config.input_height, config.input_width]
    self.resized_x = tf.image.resize_images(self.x, resize_dim)
    self.resized_y = tf.image.resize_images(self.y, resize_dim)

    self.normalized_x = normalize(self.resized_x)
    self.normalized_y = normalize(self.resized_y)

    self._build_model()
    self._build_steps()

    show_all_variables()

  def build_optim(self):
    self.refiner_step = tf.Variable(0, name='refiner_step', trainable=False)
    self.discrim_step = tf.Variable(0, name='discrim_step', trainable=False)

    def minimize(optim, loss, step, var_list):
      if self.max_grad_norm != None:
        grads_and_vars = optim.compute_gradients(loss)
        new_grads_and_vars = []
        for idx, (grad, var) in enumerate(grads_and_vars):
          if grad is not None and var in var_list:
            new_grads_and_vars.append((tf.clip_by_norm(grad, self.max_grad_norm), var))
        return optim.apply_gradients(grads_and_vars,
                                     global_step=step)
      else:
        return optim.minimize(loss, global_step=step, var_list=var_list)

    if self.task == "generative":
      #optim = tf.train.GradientDescentOptimizer(self.learning_rate)
      optim = tf.train.AdamOptimizer(self.learning_rate)
      self.refiner_optim = minimize(
          optim, self.refiner_loss, self.refiner_step, self.refiner_vars)

      #optim = tf.train.GradientDescentOptimizer(self.learning_rate)
      optim = tf.train.AdamOptimizer(self.learning_rate)
      self.discrim_optim = minimize(
          optim, self.discrim_loss, self.discrim_step, self.discrim_vars)

      #optim = tf.train.GradientDescentOptimizer(self.learning_rate)
      optim = tf.train.AdamOptimizer(self.learning_rate)
      self.discrim_optim_with_history = minimize(
          optim, self.discrim_loss_with_history, self.discrim_step, self.discrim_vars)
    elif self.task == "estimate":
      raise Exception("[!] Not implemented yet")

  def _build_model(self):
    with arg_scope([resnet_block, conv2d, max_pool2d, tanh],
                   layer_dict=self.layer_dict):
      self.R_x = self._build_refiner(self.normalized_x)
      self.denormalized_R_x = denormalize(self.R_x)

      self.D_R_x, self.D_R_x_logits = \
          self._build_discrim(self.R_x, name="D_R_x", reuse=True)
      self.D_R_x_history, self.D_R_x_history_logits = \
          self._build_discrim(self.R_x_history,
                              name="D_R_x_history", reuse=True)
      self.D_y, self.D_y_logits = \
          self._build_discrim(self.normalized_y, name="D_y")

      #self.estimate_outputs = self._build_estimation_network()
    self._build_loss()

  def _build_loss(self):
    # Refiner loss
    def fake_label(layer):
      return tf.zeros_like(layer, dtype=tf.int32)[:,:,:,0]

    def real_label(layer):
      return tf.ones_like(layer, dtype=tf.int32)[:,:,:,0]

    with tf.name_scope("refiner"):
      self.realism_loss = tf.reduce_sum(
          SE_loss(self.D_R_x_logits, real_label(self.D_R_x_logits)), [1, 2], name="realism_loss")
      self.regularization_loss = \
          self.reg_scale * tf.reduce_sum(
              tf.abs(self.R_x - self.normalized_x), [1, 2, 3],
              name="regularization_loss")

      self.refiner_loss = tf.reduce_mean(
          self.realism_loss + self.regularization_loss,
          name="refiner_loss")

      if self.debug:
        self.refiner_loss = tf.Print(
            self.refiner_loss, [self.R_x], "R_x")
        self.refiner_loss = tf.Print(
            self.refiner_loss, [self.D_R_x], "D_R_x")
        self.refiner_loss = tf.Print(
            self.refiner_loss, [self.normalized_x], "normalized_x")
        self.refiner_loss = tf.Print(
            self.refiner_loss, [self.denormalized_R_x], "denormalized_R_x")
        self.refiner_loss = tf.Print(
            self.refiner_loss, [self.regularization_loss], "reg_loss")

    self.refiner_summary = tf.summary.merge([
        tf.summary.image("synthetic_images", self.x),
        tf.summary.image("refined_images", self.denormalized_R_x),
        tf.summary.scalar("refiner/realism_loss",
                          tf.reduce_mean(self.realism_loss)),
        tf.summary.scalar("refiner/regularization_loss",
                          tf.reduce_mean(self.regularization_loss)),
        tf.summary.scalar("refiner/loss",
                          tf.reduce_mean(self.refiner_loss)),
    ])

    # Discriminator loss
    with tf.name_scope("discriminator"):
      self.refiner_d_loss = tf.reduce_sum(
          SE_loss(self.D_R_x_logits, fake_label(self.D_R_x_logits)), [1, 2],
          name="refiner_d_loss")
      self.synthetic_d_loss = tf.reduce_sum(
          SE_loss(self.D_y_logits, real_label(self.D_y_logits)), [1, 2],
          name="synthetic_d_loss")

      self.discrim_loss = tf.reduce_mean(
          self.refiner_d_loss + \
              self.synthetic_d_loss, name="discrim_loss")

      # with history
      self.refiner_d_loss_with_history = tf.reduce_sum(
          SE_loss(self.D_R_x_history_logits, fake_label(self.D_R_x_history_logits)), [1, 2],
          name="refiner_d_loss_with_history")
      self.discrim_loss_with_history = tf.reduce_mean(
          tf.concat_v2([self.refiner_d_loss, self.refiner_d_loss_with_history], axis=0) + \
              self.synthetic_d_loss, name="discrim_loss_with_history")

      if self.debug:
        self.discrim_loss_with_history = tf.Print(
            self.discrim_loss_with_history, [self.refiner_d_loss], "refiner_d_loss")
        self.discrim_loss_with_history = tf.Print(
            self.discrim_loss_with_history, [self.refiner_d_loss_with_history], "refiner_d_loss_with_history")
        self.discrim_loss_with_history = tf.Print(
            self.discrim_loss_with_history, [self.synthetic_d_loss], "synthetic_d_loss")
        self.discrim_loss_with_history = tf.Print(
            self.discrim_loss_with_history, [self.D_R_x_history_logits], "D_R_x_history_logits")
        self.discrim_loss_with_history = tf.Print(
            self.discrim_loss_with_history, [self.D_y_logits], "D_y_logits")

      self.discrim_summary = tf.summary.merge([
          tf.summary.scalar("synthetic_d_loss",
                            tf.reduce_mean(self.synthetic_d_loss)),
          tf.summary.scalar("refiner_d_loss",
                            tf.reduce_mean(self.refiner_d_loss)),
          tf.summary.scalar("discrim_loss",
                            tf.reduce_mean(self.discrim_loss)),
      ])
      self.discrim_summary_with_history = tf.summary.merge([
          tf.summary.scalar("synthetic_d_loss",
                            tf.reduce_mean(self.synthetic_d_loss)),
          tf.summary.scalar("refiner_d_loss_with_history",
                            tf.reduce_mean(self.refiner_d_loss_with_history)),
          tf.summary.scalar("discrim_loss_with_history",
                            tf.reduce_mean(self.discrim_loss_with_history)),
      ])

  def _build_steps(self):
    def run(sess, feed_dict, fetch,
            summary_op, summary_writer, output_op=None):
      if summary_writer is not None:
        fetch['summary'] = summary_op
      if output_op is not None:
        fetch['output'] = output_op

      result = sess.run(fetch, feed_dict=feed_dict)
      if result.has_key('summary'):
        summary_writer.add_summary(result['summary'], result['step'])
        summary_writer.flush()
      return result

    def train_refiner(sess, feed_dict, summary_writer=None, with_output=False):
      fetch = {
          'loss': self.refiner_loss,
          'optim': self.refiner_optim,
          'step': self.refiner_step,
      }
      return run(sess, feed_dict, fetch,
                 self.refiner_summary, summary_writer,
                 output_op=self.R_x if with_output else None)

    def test_refiner(sess, feed_dict, summary_writer=None, with_output=False):
      fetch = {
          'loss': self.refiner_loss,
          'step': self.refiner_step,
      }
      return run(sess, feed_dict, fetch,
                 self.refiner_summary, summary_writer,
                 output_op=self.R_x if with_output else None)

    def train_discrim(sess, feed_dict, summary_writer=None,
                      with_history=False, with_output=False):
      fetch = {
          'loss': self.discrim_loss_with_history,
          'optim': self.discrim_optim_with_history,
          'step': self.discrim_step,
      }
      return run(sess, feed_dict, fetch,
                 self.discrim_summary_with_history if with_history \
                     else self.discrim_summary, summary_writer,
                 output_op=self.D_R_x if with_output else None)

    def test_discrim(sess, feed_dict, summary_writer=None,
                     with_history=False, with_output=False):
      fetch = {
          'loss': self.discrim_loss,
          'step': self.discrim_step,
      }
      return run(sess, feed_dict, fetch,
                 self.discrim_summary_with_history if with_history \
                     else self.discrim_summary, summary_writer,
                 output_op=self.D_R_x if with_output else None)

    self.train_refiner = train_refiner
    self.test_refiner = test_refiner
    self.train_discrim = train_discrim
    self.test_discrim = test_discrim

  def _build_refiner(self, layer):
    with tf.variable_scope("refiner") as sc:
      layer = repeat(layer, 4, resnet_block, scope="resnet")
      layer = conv2d(layer, 1, 1, 1, 
                     activation_fn=None, scope="conv_1")
      output = tanh(layer, name="tanh")
      self.refiner_vars = tf.contrib.framework.get_variables(sc)
    return output 

  def _build_discrim(self, layer, name, reuse=False):
    with tf.variable_scope("discriminator") as sc:
      layer = conv2d(layer, 96, 3, 2, scope="conv_1", name=name)
      layer = conv2d(layer, 64, 3, 2, scope="conv_2", name=name)
      layer = max_pool2d(layer, 3, 1, scope="max_1", name=name)
      layer = conv2d(layer, 32, 3, 1, scope="conv_3", name=name)
      layer = conv2d(layer, 32, 1, 1, scope="conv_4", name=name)
      logits = conv2d(layer, 2, 1, 1, scope="conv_5", name=name)
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
