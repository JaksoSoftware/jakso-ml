import random, copy
import cv2 as cv
from .augmenter import Augmenter

__all__ = ['Rotator']

class Rotator(Augmenter):
  """
  Augmenter that rotates the SampleImages randomly based on
  the min_angle and max_angle parameters.
  """
  def __init__(
    self,
    num,
    keep_original = False,
    min_angle = -3,
    max_angle = 3,
  ):
    super().__init__(num, keep_original)

    self.min_angle = min_angle
    self.max_angle = max_angle

  def augment(self, sample):
    samples = []

    for i in range(self.num):
      angle = random.uniform(self.min_angle, self.max_angle)
      rotation_matrix = cv.getRotationMatrix2D(sample.roi_center, angle, 1)
      rotated = cv.warpAffine(sample.image, rotation_matrix, (sample.image.shape[1], sample.image.shape[0]))

      sample_copy = copy.copy(sample)
      sample_copy.image = rotated

      samples.append(sample_copy)

    return samples
