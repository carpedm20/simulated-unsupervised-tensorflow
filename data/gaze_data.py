import os
import sys
import json
import fnmatch
import tarfile
from PIL import Image
from glob import glob
from tqdm import tqdm
from six.moves import urllib

import numpy as np

from utils import loadmat, imread, imwrite

DATA_FNAME = 'gaze.npz'

def save_array_to_grayscale_image(array, path):
  Image.fromarray(array).convert('L').save(path)

def process_json_list(json_list, img):
  ldmks = [eval(s) for s in json_list]
  return np.array([(x, img.shape[0]-y, z) for (x,y,z) in ldmks])

def maybe_download_and_extract(
    config,
    data_path,
    url='http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIGaze.tar.gz'):
  if not os.path.exists(os.path.join(data_path, config.real_image_dir)):
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
  if config.max_synthetic_num < 0:
    max_synthetic_num = None
  else:
    max_synthetic_num = config.max_synthetic_num
  
  # MPIIGaze dataset
  base_path = os.path.join(data_path, '{}/Data/Normalized'.format(config.real_image_dir))
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
  synthetic_image_path_candidates = [
      config.synthetic_image_dir,
      os.path.join(data_path, config.synthetic_image_dir),
  ]
  for synthetic_image_path in synthetic_image_path_candidates:
    jpg_paths = glob(os.path.join(synthetic_image_path, '*.jpg'))
    cropped_jpg_paths = glob(os.path.join(synthetic_image_path, '*_cropped.png'))

    if len(jpg_paths) == 0:
      print("[!] No images in ./{}. Skip.".format(synthetic_image_path))
      continue
    else:
      print("[!] Found images in ./{}.".format(synthetic_image_path))
      if len(cropped_jpg_paths) != len(jpg_paths):
        json_paths = glob(os.path.join(
            data_path, '{}/*.json'.format(config.synthetic_image_dir)))

        assert len(jpg_paths) >= max_synthetic_num, \
            "[!] # of synthetic data ({}) is smaller than max_synthetic_num ({})". \
                format(len(jpg_paths), max_synthetic_num)

        json_paths = json_paths[:max_synthetic_num]
        for json_path in tqdm(json_paths):
          jpg_path = json_path.replace('json', 'jpg')

          if not os.path.exists(jpg_path):
            continue

          with open(json_path) as json_f:
            img = imread(jpg_path)
            j = json.loads(json_f.read())

            key = "interior_margin_2d"
            j[key] = process_json_list(j[key], img)

            x_min, x_max = int(min(j[key][:,0])), int(max(j[key][:,0]))
            y_min, y_max = int(min(j[key][:,1])), int(max(j[key][:,1]))

            x_center, y_center = (x_min + x_max)/2, (y_min + y_max)/2

            cropped_img = img[y_center-42: y_center+42, x_center-70:x_center+70]
            img_path = jpg_path.replace(".jpg", "_cropped.png")

            save_array_to_grayscale_image(cropped_img, img_path)

      jpg_paths = glob(os.path.join(synthetic_image_path, '*.jpg'))
      cropped_jpg_paths = glob(os.path.join(synthetic_image_path, '*_cropped.png'))

      print("[*] # of synthetic data: {}, # of cropped_data: {}". \
          format(len(jpg_paths), len(cropped_jpg_paths)))
      print("[*] Finished preprocessing synthetic `gaze` data.")

      return synthetic_image_path

  raise Exception("[!] Failed to found proper synthetic_image_path in {}" \
      .format(synthetic_image_path_candidates))

def load(config, data_path, sample_path, rng):
  if not os.path.exists(data_path):
    print('creating folder', data_path)
    os.makedirs(data_path)

  maybe_download_and_extract(config, data_path)
  synthetic_image_path = maybe_preprocess(config, data_path, sample_path)

  gaze_data = np.load(os.path.join(data_path, DATA_FNAME))
  real_data = gaze_data['real']

  if not os.path.exists(sample_path):
    os.makedirs(sample_path)

  print("[*] Save samples images in {}".format(data_path))
  random_idxs = rng.choice(len(real_data), 100)
  for idx, random_idx in enumerate(random_idxs):
    image_path = os.path.join(sample_path, "real_{}.png".format(idx))
    imwrite(image_path, real_data[random_idx])

  return real_data, synthetic_image_path

class DataLoader(object):
  def __init__(self, config, rng=None):
    self.rng = np.random.RandomState(1) if rng is None else rng

    self.data_path = os.path.join(config.data_dir, 'gaze')
    self.sample_path = os.path.join(self.data_path, config.sample_dir)
    self.batch_size = config.batch_size
    self.debug = config.debug

    self.real_data, synthetic_image_path = load(config, self.data_path, self.sample_path, rng)

    self.synthetic_data_paths = np.array(glob(os.path.join(synthetic_image_path, '*_cropped.png')))
    self.synthetic_data_dims = list(imread(self.synthetic_data_paths[0]).shape) + [1]

    self.synthetic_data_paths.sort()

    if np.rank(self.real_data) == 3:
      self.real_data = np.expand_dims(self.real_data, -1)
    
    self.real_p = 0

  def get_observation_size(self):
    return self.real_data.shape[1:]

  def get_num_labels(self):
    return np.amax(self.labels) + 1

  def reset(self):
    self.real_p = 0

  def __iter__(self):
    return self

  def __next__(self, n=None):
    """ n is the number of examples to fetch """
    if n is None: n = self.batch_size

    if self.real_p == 0:
      inds = self.rng.permutation(self.real_data.shape[0])
      self.real_data = self.real_data[inds]

    if self.real_p + n > self.real_data.shape[0]:
      self.reset()

    x = self.real_data[self.real_p : self.real_p + n]
    self.real_p += self.batch_size

    return x

  next = __next__
