import copy
import cv2 as cv
from .augmenter import Augmenter

class HorizontalFlipper(Augmenter):
  '''
  Augmenter that flips the image horizontally.
  '''
  def __init__(self, num = 1, keep_original = True):
    super().__init__(num, keep_original)

  def augment(self, sample):
    sample_copy = copy.copy(sample)
    sample_copy.image = cv.flip(sample_copy.image, 1)

    return [sample_copy]
