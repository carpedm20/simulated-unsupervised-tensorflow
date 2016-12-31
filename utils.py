import os
import numpy as np
from datetime import datetime

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

def process_json_list(json_list, img):
  ldmks = [eval(s) for s in json_list]
  return np.array([(x, img.shape[0]-y, z) for (x,y,z) in ldmks])
