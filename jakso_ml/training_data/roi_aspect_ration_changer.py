import copy
from .augmenter import Augmenter
from ..utils import round_tuple

class RoiAspectRatioChanger(Augmenter):
  '''
  Augmenter that fixes the aspect ratio of the ROI to the specified
  aspect_ratio (width / height).

  If the aspect ratio needs to be changed, it is always done so that
  the ROI gets larger, and not smaller.
  '''
  def __init__(self, aspect_ratio, **kwargs):
    super().__init__(**kwargs)
    self.aspect_ratio = aspect_ratio

  def augment(self, sample):
    x, y, w, h = sample.roi

    if w / h > self.aspect_ratio:
      new_w = w
      new_h = w / self.aspect_ratio
    else:
      new_w = h * self.aspect_ratio
      new_h = h

    new_x = x + (w - new_w) / 2
    new_y = y + (h - new_h) / 2

    sample_copy = copy.copy(sample)
    sample_copy.roi = round_tuple((new_x, new_y, new_w, new_h))

    return sample_copy
