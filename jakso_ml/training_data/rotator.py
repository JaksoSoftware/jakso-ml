import random, copy
import cv2 as cv
from .augmenter import Augmenter

class Rotator(Augmenter):
  '''
  Augmenter that rotates the SampleImages randomly based on
  the min_angle and max_angle parameters.
  '''
  def __init__(
    self,
    min_angle,
    max_angle,
    **kwargs
  ):
    super().__init__(**kwargs)

    self.min_angle = min_angle
    self.max_angle = max_angle

  def augment(self, sample):
    im_h, im_w, _  = sample.image.shape

    angle = random.uniform(self.min_angle, self.max_angle)
    rotation_matrix = cv.getRotationMatrix2D(sample.roi_center, angle, 1)
    rotated = cv.warpAffine(sample.image, rotation_matrix, (im_w, im_h))

    sample_copy = copy.copy(sample)
    sample_copy.image = rotated

    return sample_copy
