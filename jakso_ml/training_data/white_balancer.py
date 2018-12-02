import random, copy
import cv2 as cv
import numpy as np
from scipy import interpolate
from .augmenter import Augmenter

class WhiteBalancer(Augmenter):
  '''
  Augmenter that randomly changes the white balance of the SampleImages.
  '''
  def __init__(
    self,
    num = 1,
    keep_original = False,
    min_red_rand = -0.1,
    max_red_rand = 0.2,
    min_blue_rand = -0.1,
    max_blue_rand = 0.1
  ):
    super().__init__(num, keep_original)

    self.min_red_rand = min_red_rand
    self.max_red_rand = max_red_rand
    self.min_blue_rand = min_blue_rand
    self.max_blue_rand = max_blue_rand

  def augment(self, sample):
    samples = []

    for i in range(self.num):
      sample_copy = copy.deepcopy(sample)
      b, g, r = cv.split(sample_copy.image)

      rand_b = 128 * random.uniform(1 + self.min_blue_rand, 1 + self.max_blue_rand)
      rand_r = 0

      if rand_b < 1:
        rand_r = 128 * random.uniform(1, 1 + self.max_red_rand)
      else:
        rand_r = 128 * random.uniform(1 + self.min_red_rand, 1)

      lut_b = self._create_lut(rand_b)
      lut_r = self._create_lut(rand_r)

      b = cv.LUT(b, lut_b)
      r = cv.LUT(r, lut_r)

      sample_copy.image = cv.merge((b, g, r))
      samples.append(sample_copy)

    return samples

  def _create_lut(self, center):
    tck = interpolate.splrep([0, 128, 256], [0, center, 256], k = 2)
    lut = np.rint(interpolate.splev(range(256), tck, der = 0))
    lut = np.uint8(lut)
    return lut
