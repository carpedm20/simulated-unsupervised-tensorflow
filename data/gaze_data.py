import os
import sys
import json
import fnmatch
import tarfile
from glob import glob
from tqdm import tqdm
from six.moves import urllib

import numpy as np

from utils import loadmat, imread, imwrite, process_json_list

DATA_FNAME = 'gaze.npz'

def maybe_download_and_extract(
    config,
    data_path,
    url='http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIGaze.tar.gz'):
  if not os.path.exists(os.path.join(data_path, config.mpiigaze_dir)):
    if not os.path.exists(data_path):
      os.makedirs(data_path)

    filename = os.path.basename(url)
    filepath = os.path.join(data_path, filename)

    if not os.path.exists(filepath):
      def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

      filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
      statinfo = os.stat(filepath)
      print('\nSuccessfully downloaded {} {} bytes.'.format(filename, statinfo.st_size))
      tarfile.open(filepath, 'r:gz').extractall(data_path)

def maybe_preprocess(config, data_path, sample_path=None):
  max_synthetic_num = config.max_synthetic_num
  
  # MPIIGaze dataset
  base_path = os.path.join(data_path, '{}/Data/Normalized'.format(config.mpiigaze_dir))
  npz_path = os.path.join(data_path, DATA_FNAME)

  if not os.path.exists(npz_path):
    mat_paths = []
    for root, dirnames, filenames in os.walk(base_path):
      for filename in fnmatch.filter(filenames, '*.mat'):
        mat_paths.append(os.path.join(root, filename))

    print("[*] Preprocessing real `gaze` data...")

    real_images =[]
    for mat_path in tqdm(mat_paths):
      mat = loadmat(mat_path)
      # Left eye (batch_size, height, width)
      real_images.extend(mat['data'][0][0][0][0][0][1])
      # Right eye
      real_images.extend(mat['data'][0][0][1][0][0][1])

    real_data = np.stack(real_images, axis=0)
    np.savez(npz_path, real=real_data)

  # UnityEyes dataset
  cropped_jpg_paths = glob(os.path.join(data_path, '{}/*_cropped.png'.format(config.unityeye_dir)))
  jpg_paths = glob(os.path.join(data_path, '{}/*.jpg'.format(config.unityeye_dir)))

  if len(cropped_jpg_paths) != len(jpg_paths):
    json_paths = glob(os.path.join(data_path, '{}/*.json'.format(config.unityeye_dir)))

    assert len(jpg_paths) >= max_synthetic_num, \
        "[!] # of synthetic data ({}) is smaller than max_synthetic_num ({})". \
            format(len(jpg_paths), max_synthetic_num)

    jpg_paths = jpg_paths[:max_synthetic_num]
    json_paths = json_paths[:max_synthetic_num]

    print("[*] Preprocessing synthetic `gaze` data...")
    for (jpg_path, json_path) in tqdm(zip(jpg_paths, json_paths)):
      with open(json_path) as json_f:
        img = imread(jpg_path)
        j = json.loads(json_f.read())

        for key in ["interior_margin_2d"]: #, "caruncle_2d", "iris_2d"]:
          j[key] = process_json_list(j[key], img)

          x_min, x_max = int(min(j[key][:,0])), int(max(j[key][:,0]))
          y_min, y_max = int(min(j[key][:,1])), int(max(j[key][:,1]))

        x_center, y_center = (x_min + x_max)/2, (y_min + y_max)/2
        imwrite(jpg_path.replace(".jpg", "_cropped.png"), img[y_center-42: y_center+42, x_center-70:x_center+70])

def load(config, data_path, sample_path):
  if not os.path.exists(data_path):
    print('creating folder', data_path)
    os.makedirs(data_path)

  maybe_download_and_extract(config, data_path)
  maybe_preprocess(config, data_path, sample_path)

  gaze_data = np.load(os.path.join(data_path, DATA_FNAME))
  real_data = gaze_data['real']

  if config.debug:
    if not os.path.exists(sample_path):
      os.makedirs(sample_path)

    print("[*] Save samples images in {}".format(data_path))
    for idx in range(100):
      image_path = os.path.join(sample_path, "real_{}.png".format(idx))
      imwrite(image_path, real_data[idx])

  return real_data

class DataLoader(object):
  def __init__(self, config, rng=None):
    self.data_path = os.path.join(config.data_dir, 'gaze')
    self.sample_path = os.path.join(self.data_path, config.sample_dir)
    self.batch_size = config.batch_size
    self.debug = config.debug

    self.data = load(config, self.data_path, self.sample_path)
    if np.rank(self.data) == 3:
      self.data = np.expand_dims(self.data, -1)
    
    self.p = 0 # pointer to where we are in iteration
    self.rng = np.random.RandomState(1) if rng is None else rng

  def get_observation_size(self):
    return self.data.shape[1:]

  def get_num_labels(self):
    return np.amax(self.labels) + 1

  def reset(self):
    self.p = 0

  def __iter__(self):
    return self

  def __next__(self, n=None):
    """ n is the number of examples to fetch """
    if n is None: n = self.batch_size

    # on first iteration lazily permute all data
    if self.p == 0:
      inds = self.rng.permutation(self.data.shape[0])
      self.data = self.data[inds]

    # on last iteration reset the counter and raise StopIteration
    if self.p + n > self.data.shape[0]:
      self.reset() # reset for next time we get called
      raise StopIteration

    # on intermediate iterations fetch the next batch
    x = self.data[self.p : self.p + n]
    self.p += self.batch_size

    return x

  next = __next__
