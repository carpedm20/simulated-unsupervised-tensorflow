import os
import numpy as np
from tqdm import trange
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope

from model import Model
from buffer import Buffer
import data.gaze_data as gaze_data
import data.hand_data as hand_data
from utils import imwrite, imread, img_tile

class Trainer(object):
  def __init__(self, config, rng):
    self.config = config
    self.task = config.task
    self.model_dir = config.model_dir
    self.gpu_memory_fraction = config.gpu_memory_fraction

    self.log_step = config.log_step
    self.max_step = config.max_step

    self.K_d = config.K_d
    self.K_g = config.K_g
    self.initial_K_d = config.initial_K_d
    self.initial_K_g = config.initial_K_g
    self.checkpoint_secs = config.checkpoint_secs

    DataLoader = {
        'gaze': gaze_data.DataLoader,
        'hand': hand_data.DataLoader,
    }[config.data_set]
    self.data_loader = DataLoader(config, rng=rng)

    self.model = Model(config, self.data_loader)
    self.history_buffer = Buffer(config, rng)

    self.summary_ops = {
        'test_synthetic_images': {
            'summary': tf.summary.image("test_synthetic_images",
                                        self.model.resized_x,
                                        max_outputs=config.max_image_summary),
            'output': self.model.resized_x,
        },
        'test_refined_images': {
            'summary': tf.summary.image("test_refined_images",
                                        self.model.denormalized_R_x,
                                        max_outputs=config.max_image_summary),
            'output': self.model.denormalized_R_x,
        }
    }

    self.saver = tf.train.Saver()
    self.summary_writer = tf.summary.FileWriter(self.model_dir)

    sv = tf.train.Supervisor(logdir=self.model_dir,
                             is_chief=True,
                             saver=self.saver,
                             summary_op=None,
                             summary_writer=self.summary_writer,
                             save_summaries_secs=300,
                             save_model_secs=self.checkpoint_secs,
                             global_step=self.model.discrim_step)

    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=self.gpu_memory_fraction,
        allow_growth=True) # seems to be not working
    sess_config = tf.ConfigProto(allow_soft_placement=True,
                                 gpu_options=gpu_options)

    self.sess = sv.prepare_or_wait_for_session(config=sess_config)

  def train(self):
    print("[*] Training starts...")
    self._summary_writer = None

    test_samples = np.expand_dims(np.stack(
        [imread(path) for path in \
            self.data_loader.synthetic_data_paths[:self.config.max_image_summary]]
    ), -1)

    def train_refiner():
      feed_dict = {
        self.model.synthetic_batch_size: self.data_loader.batch_size,
      }
      res = self.model.train_refiner(
          self.sess, feed_dict, self._summary_writer, with_output=True)
      self.history_buffer.push(res['output'])
      self._summary_writer = self._get_summary_writer(res)

      if res['step'] % self.log_step == 0:
        feed_dict = {
            self.model.x: test_samples,
        }
        self._inject_summary(
          'test_refined_images', feed_dict, res['step'])

        if res['step'] / float(self.log_step) == 1.:
          self._inject_summary(
              'test_synthetic_images', feed_dict, res['step'])

    def train_discrim():
      feed_dict = {
        self.model.synthetic_batch_size: self.data_loader.batch_size/2,
        self.model.R_x_history: self.history_buffer.sample(),
        self.model.y: self.data_loader.next(),
      }
      res = self.model.train_discrim(
          self.sess, feed_dict, self._summary_writer, with_history=True, with_output=False)
      self._summary_writer = self._get_summary_writer(res)

    for k in trange(self.initial_K_g, desc="Train refiner"):
      train_refiner()

    for k in trange(self.initial_K_d, desc="Train discrim"):
      train_discrim()

    for step in trange(self.max_step, desc="Train both"):
      for k in xrange(self.K_g):
        train_refiner()

      for k in xrange(self.K_d):
        train_discrim()

  def test(self):
    batch_size = self.data_loader.batch_size
    for idx in trange(len(self.data_loader.synthetic_data_paths) / batch_size):
      feed_dict = {
        self.model.synthetic_batch_size: batch_size,
      }
      res = self.model.test_refiner(
          self.sess, feed_dict, self._summary_writer, with_output=True)
      import ipdb; ipdb.set_trace() 
      x = 123

  def _inject_summary(self, tag, feed_dict, step):
    summaries = self.sess.run(self.summary_ops[tag], feed_dict)
    self.summary_writer.add_summary(summaries['summary'], step)

    path = os.path.join(
        self.config.sample_dir, self.config.model_name, "{}.png".format(step))
    imwrite(path, img_tile(summaries['output'], tile_shape=[8, 8])[:,:,0])

  def _get_summary_writer(self, result):
    if result['step'] % self.log_step == 0:
      return self.summary_writer
    else:
      return None
