import os
import sys
import time
import json
import argparse
import numpy as np
import tensorflow as tf

from trainer import Trainer
from config import get_config
from utils import prepare_dirs

config = None

def main(_):
  prepare_dirs(config)

  rng = np.random.RandomState(config.random_seed)
  tf.set_random_seed(config.random_seed)

  trainer = Trainer(config, rng)

  if config.is_train:
    trainer.train()
  else:
    if not config.load_path:
      raise Exception("[!] You should specify `load_path` to load a pretrained model")
    trainer.test()

if __name__ == "__main__":
  config, unparsed = get_config()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
