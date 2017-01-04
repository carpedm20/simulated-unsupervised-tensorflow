import os
import numpy as np
from datetime import datetime

import tensorflow as tf

try:
  import scipy.misc
  imread = scipy.misc.imread
  imresize = scipy.misc.imresize
  imwrite = scipy.misc.imsave
except:
  import cv2
  imread = cv2.imread
  imresize = cv2.resize
  imwrite = cv2.imwrite

import scipy.io as sio
loadmat = sio.loadmat

def prepare_dirs(config):
  if config.load_path:
    config.model_dir = os.path.join(
        config.log_dir, "{}_{}".format(config.task, config.load_path))
  else:
    config.model_dir = os.path.join(
        config.log_dir, "{}_{}".format(config.task, get_time()))

  for path in [config.log_dir, config.data_dir]:
    if not os.path.exists(path):
      os.makedirs(path)

def get_time():
  return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def show_all_variables():
  print("")
  total_count = 0
  for idx, op in enumerate(tf.trainable_variables()):
    shape = op.get_shape()
    count = np.prod(shape)
    print("[%2d] %s %s = %s" % (idx, op.name, shape, "{:,}".format(int(count))))
    total_count += int(count)
  print("=" * 40)
  print("[Total] variable size: %s" % "{:,}".format(total_count))
  print("=" * 40)
  print("")
