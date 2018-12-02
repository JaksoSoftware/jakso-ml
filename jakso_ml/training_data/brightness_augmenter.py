import random, copy
import numpy as np
from .augmenter import Augmenter

__all__ = ['BrightnessAugmenter']

class BrightnessAugmenter(Augmenter):
  def __init__(
    self,
    num = 1,
    keep_original = False,
    min = 50,
    max = 255
  ):
    super().__init__(num, keep_original)

    self.min = min
    self.max = max

  def augment(self, sample):
    minim = np.min(sample.image)
    maxim = np.max(sample.image)

    sample_copy = copy.deepcopy(sample)

    brightness = round(
      random.uniform(
        min(self.min - minim, 0),
        self.max - maxim
      )
    )

    sample_copy.image = sample_copy.image + brightness
    return [sample_copy]
