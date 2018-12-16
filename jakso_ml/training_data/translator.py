import random, copy
from .augmenter import Augmenter
from ..utils import area, intersection, round_tuple

class Translator(Augmenter):
  '''
  Augmenter that translates the SampleImages randomly based on
  the min/max_trans_x and min/max_trans_y parameters.
  '''
  def __init__(
    self,
    min_trans_x,
    max_trans_x,
    min_trans_y,
    max_trans_y,
    **kwargs
  ):
    super().__init__(**kwargs)

    self.min_trans_x = min_trans_x
    self.max_trans_x = max_trans_x
    self.min_trans_y = min_trans_y
    self.max_trans_y = max_trans_y

  def augment(self, sample):
    im_h, im_w, _ = sample.image.shape
    im_bounds = (0, 0, im_w, im_h)
    x, y, w, h = sample.roi

    trans_x = w * random.uniform(self.min_trans_x, self.max_trans_x)
    trans_y = h * random.uniform(self.min_trans_y, self.max_trans_y)

    tx = x + trans_x
    ty = y + trans_y

    sample_copy = copy.copy(sample)
    sample_copy.roi = round_tuple((tx, ty, w, h))

    # Skip the sample if the translated ROI goes out of bounds
    if area(intersection(sample_copy.roi, im_bounds)) == area(sample_copy.roi):
      return sample_copy
    else:
      print('skipping translate', sample_copy.roi)
      return None
