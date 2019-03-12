import random
from .processor import Processor

class Augmenter(Processor):
  '''
  Augmenter is a Processor that augments the input SampleImage.
  '''
  def __init__(self, num = 1, keep_original = False, augment_propability = 1):
    # num < 1 means a propability that this augmenter creates outputs.
    self.augment_propability = num
    self.num = int(max(num, 1))
    self.keep_original = keep_original

  def process(self, sample):
    if random.uniform(0, 1) > self.augment_propability:
      return [sample]

    augmented_samples = []

    for _ in range(self.num):
      augmented_sample = self.augment(sample)

      if augmented_sample != None:
        augmented_samples.append(augmented_sample)

    if self.keep_original:
      augmented_samples.append(sample)

    return augmented_samples

  def augment(self, sample):
    return sample
