from tqdm import tqdm

import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope

from network import Network

class Model(object):
  def __init__(self, config):
    self.K_d = config.K_d
    self.K_g = config.K_g

    self.network = Network(config)

  def train(self):
    for step in range(self.max_step):
      for k in range(self.K_g):
        pass

      for k in range(self.K_d):
        pass
