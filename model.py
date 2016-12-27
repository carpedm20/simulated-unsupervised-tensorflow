import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope

from network import Network

class Model(object):
  def __init__(self, config):
    self.network = Network(config)
    pass
