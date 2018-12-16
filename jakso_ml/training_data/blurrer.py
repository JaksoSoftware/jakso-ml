import random, copy
import cv2 as cv
from .augmenter import Augmenter

class Blurrer(Augmenter):
  '''
  Blurs the input image.
  '''
  def __init__(
    self,
    blur_radii,
    **kwargs
  ):
    super().__init__(**kwargs)
    self.blur_radii = blur_radii

  def augment(self, sample):
    radius = random.choice(self.blur_radii)

    sample_copy = copy.deepcopy(sample)
    sample_copy.image = cv.GaussianBlur(sample_copy.image, (radius, radius), 0)

    return sample_copy
