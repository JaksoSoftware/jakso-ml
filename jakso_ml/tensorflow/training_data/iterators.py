import os, random
import cv2 as cv
from tensorflow import keras

__all__ = ['ImageFileIterator']

class ImageFileIterator(keras.preprocessing.image.Iterator):
  '''
  Keras image iterators that iterates over a list of image files.
  '''
  def __init__(
    self,
    file_paths,
    batch_size,
    parse_output,
    create_input_tensor,
    create_output_tensor,
    max_cache_size_megabytes = 1.5 * 1024
  ):
    super().__init__(
      n = len(file_paths),
      batch_size = batch_size,
      shuffle = True,
      seed = None
    )

    self.file_paths = file_paths
    self.parse_output = parse_output
    self.create_input_tensor = create_input_tensor
    self.create_output_tensor = create_output_tensor
    self.max_cache_size_bytes = max_cache_size_megabytes * 1024 * 1024
    self.cache_size_bytes = 0
    self.cache = {}

  def _get_batches_of_transformed_samples(self, indexes):
    inputs = []
    outputs = []

    for index in indexes:
      file_path = self.file_paths[index]

      if file_path in self.cache:
        image = self.cache[file_path]
      else:
        image = cv.imread(file_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image_size = image.shape[0] * image.shape[1] * image.shape[2]

        if self.cache_size_bytes + image_size <= self.max_cache_size_bytes:
          self.cache[file_path] = image
          self.cache_size_bytes += image_size

      output = self.parse_output(file_path, image)

      inputs.append(image)
      outputs.append(output)

    x = self.create_input_tensor(inputs)
    y = self.create_output_tensor(outputs)

    return x, y
