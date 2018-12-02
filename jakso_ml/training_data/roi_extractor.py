import copy
from .augmenter import Augmenter

class RoiExtractor(Augmenter):
  '''
  Augmenter that extracts the ROI from the image and returns
  that as a new SampleImage.
  '''
  def augment(self, sample):
    _, _, w, h = sample.roi
    sample_copy = copy.copy(sample)

    sample_copy.image = sample.roi_image
    sample_copy.roi = (0, 0, w, h)

    return [sample_copy]
