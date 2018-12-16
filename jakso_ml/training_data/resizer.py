import copy
import cv2 as cv
from .augmenter import Augmenter

class Resizer(Augmenter):
  '''
  Resizes the input image.
  '''
  def __init__(self, size, **kwargs):
    super().__init__(**kwargs)
    self.size = size

  def augment(self, sample):
    sample_copy = copy.copy(sample)
    sample_copy.image = cv.resize(sample_copy.image, self.size)

    return sample_copy
