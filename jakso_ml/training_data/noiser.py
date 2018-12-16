import random, copy
import numpy as np
from .augmenter import Augmenter

class Noiser(Augmenter):
  def __init__(
    self,
    gaussian_scale,
    salt_vs_pepper,
    salt_pepper_amount,
    **kwargs
  ):
    super().__init__(**kwargs)

    self.gaussian_scale = gaussian_scale
    self.salt_vs_pepper = salt_vs_pepper
    self.salt_pepper_amount = salt_pepper_amount

  def augment(self, sample):
    sample_copy = copy.deepcopy(sample)

    scale = random.uniform(self.gaussian_scale / 5, self.gaussian_scale)
    noise = np.random.normal(0, scale, sample_copy.image.shape)
    image = np.float64(sample_copy.image) + noise

    salt_pepper_amount = random.uniform(self.salt_pepper_amount / 5, self.salt_pepper_amount)
    num_salt = np.ceil(salt_pepper_amount * image.size * self.salt_vs_pepper)
    num_pepper = np.ceil(salt_pepper_amount * image.size * (1.0 - self.salt_vs_pepper))

    channels_start = random.randint(0, 3)
    channels_end = random.randint(channels_start, 3)

    # Add Salt noise
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    image[coords[0], coords[1], channels_start:channels_end] = 255

    # Add Pepper noise
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    image[coords[0], coords[1], channels_start:channels_end] = 0

    sample_copy.image = np.uint8(np.clip(image, 0, 255))

    return sample_copy
