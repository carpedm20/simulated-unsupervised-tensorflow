import sys
import numpy as np
import tensorflow as tf

from trainer import Trainer
from config import get_config
from utils import prepare_dirs, save_config

config = None

def main(_):
  prepare_dirs(config)

  rng = np.random.RandomState(config.random_seed)
  tf.set_random_seed(config.random_seed)

  trainer = Trainer(config, rng)
  save_config(config.model_dir, config)

  if config.is_train:
    trainer.train()
  else:
    if not config.load_path:
      raise Exception("[!] You should specify `load_path` to load a pretrained model")
    trainer.test()

if __name__ == "__main__":
  config, unparsed = get_config()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
