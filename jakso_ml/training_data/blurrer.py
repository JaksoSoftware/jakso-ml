import random, copy
import cv2 as cv
from .augmenter import Augmenter

__all__ = ['Blurrer']

class Blurrer(Augmenter):
  """
  Blurs the input image.
  """
  def __init__(
    self,
    num = 1,
    keep_original = False,
    blur_radii = [3]
  ):
    super().__init__(num, keep_original)
    self.blur_radii = blur_radii

  def augment(self, sample):
    radius = random.choice(self.blur_radii)

    sample_copy = copy.deepcopy(sample)
    sample_copy.image = cv.GaussianBlur(sample_copy.image, (radius, radius), 0)

    return [sample_copy]
