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
