import random, copy
from .augmenter import Augmenter
from ..utils import area, intersection, round_tuple

__all__ = ['Scaler']

class Scaler(Augmenter):
  """
  Augmenter that scales the SampleImages randomly based on
  the min_scale and max_scale parameters.
  """
  def __init__(
    self,
    num,
    keep_original = False,
    min_scale = 0.8,
    max_scale = 1.3
  ):
    super().__init__(num, keep_original)

    self.min_scale = min_scale
    self.max_scale = max_scale

  def augment(self, sample):
    samples = []

    im_h, im_w, _ = sample.image.shape
    im_bounds = (0, 0, im_w, im_h)

    for i in range(self.num):
      x, y, w, h = sample.roi

      scale = random.uniform(self.min_scale, self.max_scale)
      sw = scale * w
      sh = scale * h
      sx = x + (w - sw) / 2
      sy = y + (h - sh) / 2

      sample_copy = copy.copy(sample)
      sample_copy.roi = round_tuple((sx, sy, sw, sh))

      # Skip the sample if the scaled ROI goes out of bounds
      if area(intersection(sample_copy.roi, im_bounds)) == area(sample_copy.roi):
        samples.append(sample_copy)
      else:
        print('skipping scale', sample_copy.roi)

    return samples
