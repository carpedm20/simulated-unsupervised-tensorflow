import os
import json
import numpy as np
from datetime import datetime

import tensorflow as tf
import tensorflow.contrib.slim as slim

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
    if config.load_path.startswith(config.task):
      config.model_name = config.load_path
    else:
      config.model_name = "{}_{}".format(config.task, config.load_path)
  else:
    config.model_name = "{}_{}".format(config.task, get_time())

  config.model_dir = os.path.join(config.log_dir, config.model_name)
  config.sample_model_dir = os.path.join(config.sample_dir, config.model_name)
  config.output_model_dir = os.path.join(config.output_dir, config.model_name)

  for path in [config.log_dir, config.data_dir,
               config.sample_dir, config.sample_model_dir,
               config.output_dir, config.output_model_dir]:
    if not os.path.exists(path):
      os.makedirs(path)

def get_time():
  return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def show_all_variables():
  model_vars = tf.trainable_variables()
  slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def img_tile(imgs, aspect_ratio=1.0, tile_shape=None, border=1,
             border_color=0):
  ''' from https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/plotting.py
  '''

  if imgs.ndim != 3 and imgs.ndim != 4:
    raise ValueError('imgs has wrong number of dimensions.')
  n_imgs = imgs.shape[0]

  # Grid shape
  img_shape = np.array(imgs.shape[1:3])
  if tile_shape is None:
    img_aspect_ratio = img_shape[1] / float(img_shape[0])
    aspect_ratio *= img_aspect_ratio
    tile_height = int(np.ceil(np.sqrt(n_imgs * aspect_ratio)))
    tile_width = int(np.ceil(np.sqrt(n_imgs / aspect_ratio)))
    grid_shape = np.array((tile_height, tile_width))
  else:
    assert len(tile_shape) == 2
    grid_shape = np.array(tile_shape)

  # Tile image shape
  tile_img_shape = np.array(imgs.shape[1:])
  tile_img_shape[:2] = (img_shape[:2] + border) * grid_shape[:2] - border

  # Assemble tile image
  tile_img = np.empty(tile_img_shape)
  tile_img[:] = border_color
  for i in range(grid_shape[0]):
    for j in range(grid_shape[1]):
      img_idx = j + i*grid_shape[1]
      if img_idx >= n_imgs:
        # No more images - stop filling out the grid.
        break
      img = imgs[img_idx]
      yoff = (img_shape[0] + border) * i
      xoff = (img_shape[1] + border) * j
      tile_img[yoff:yoff+img_shape[0], xoff:xoff+img_shape[1], ...] = img

  return tile_img

def save_config(model_dir, config):
  param_path = os.path.join(model_dir, "params.json")

  print("[*] MODEL dir: %s" % model_dir)
  print("[*] PARAM path: %s" % param_path)

  with open(param_path, 'w') as fp:
    json.dump(config.__dict__, fp,  indent=4, sort_keys=True)
