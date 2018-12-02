import random
from .processor import Processor

__all__ = ['Augmenter']

class Augmenter(Processor):
  """
  Augmenter is a Processor that augments the input SampleImage.
  """
  def __init__(self, num = 1, keep_original = False):
    # num < 1 means a propability that this augmenter creates outputs.
    self.augment_propability = num
    self.num = int(max(num, 1))
    self.keep_original = keep_original

  def process(self, sample):
    if random.uniform(0, 1) > self.augment_propability:
      return [sample]

    augmented = self.augment(sample)

    if self.keep_original:
      augmented.append(sample)

    return augmented

  def augment(self, sample):
    return [sample]
