import os
import sys
import time
import json
import argparse

import numpy as np
import tensorflow as tf

from model import Model
from config import get_config

import data.gaze_data as gaze_data
import data.hand_data as hand_data

config = None

def main(_):
  rng = np.random.RandomState(config.random_seed)
  tf.set_random_seed(config.random_seed)

  DataLoader = {
      'gaze': gaze_data.DataLoader,
      'hand': hand_data.DataLoader,
  }[config.data_set]

  model = Model(config)
  data_loader = DataLoader(config.data_dir, config.batch_size,
                           config.debug, rng=rng)

if __name__ == "__main__":
  config, unparsed = get_config()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
