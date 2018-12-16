import random, copy
import numpy as np
from .augmenter import Augmenter

class BrightnessAugmenter(Augmenter):
  def __init__(
    self,
    min,
    max,
    **kwargs
  ):
    super().__init__(**kwargs)

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
    return sample_copy
