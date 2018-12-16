import copy
import cv2 as cv
from .augmenter import Augmenter

class VerticalFlipper(Augmenter):
  '''
  Augmenter that flips the image vertically.
  '''
  def augment(self, sample):
    sample_copy = copy.copy(sample)
    sample_copy.image = cv.flip(sample_copy.image, 0)

    return sample_copy
