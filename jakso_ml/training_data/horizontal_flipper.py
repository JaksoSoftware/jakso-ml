import copy
import cv2 as cv
from .augmenter import Augmenter

class HorizontalFlipper(Augmenter):
  '''
  Augmenter that flips the image horizontally.
  '''
  def augment(self, sample):
    sample_copy = copy.copy(sample)
    sample_copy.image = cv.flip(sample_copy.image, 1)

    return sample_copy
