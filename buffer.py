import numpy as np

class Buffer(object):
  def __init__(self, config):
    self.buffer_size = config.buffer_size
    self.batch_size = config.batch_size

    image_dims = [
        config.input_height,
        config.input_width,
        config.input_channel,
    ]

    self.idx = 0
    self.data = np.zeros([self.buffer_size] + image_dims)

  def push(self, batches):
    batch_size = len(batches)
    if self.idx+batch_size > self.buffer_size:
      raise Exception("[!] Can't push batches to buffer more")

    self.data[self.idx:self.idx+batch_size] = self.data
    self.idx += batch_size

  def update(self, batches):
    pass
