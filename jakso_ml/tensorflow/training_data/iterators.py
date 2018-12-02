import os, random
import cv2 as cv
from tensorflow import keras

__all__ = ['create_image_iterators_for_folder', 'ImageIterator']

def create_image_iterators_for_folder(
  data_dir,
  test_portion,
  batch_size,
  parse_output,
  create_input_tensor,
  create_output_tensor
):
  '''
  Creates a Keras Iterators for reading the training and testing samples from
  a single folder. The test samples are selected randomly.
  '''
  files = os.listdir(data_dir)
  random.shuffle(files)

  num_test = round(test_portion * len(files))
  test_files = files[0:num_test]
  train_files = files[num_test:]

  test_iter = ImageIterator(
    files = test_files,
    files_dir = data_dir,
    batch_size = batch_size,
    parse_output = parse_output,
    create_input_tensor = create_input_tensor,
    create_output_tensor = create_output_tensor
  )

  train_iter = ImageIterator(
    files = train_files,
    files_dir = data_dir,
    batch_size = batch_size,
    parse_output = parse_output,
    create_input_tensor = create_input_tensor,
    create_output_tensor = create_output_tensor
  )

  return train_iter, test_iter

class ImageIterator(keras.preprocessing.image.Iterator):
  def __init__(
    self,
    files,
    files_dir,
    batch_size,
    parse_output,
    create_input_tensor,
    create_output_tensor
  ):
    super().__init__(
      n = len(files),
      batch_size = batch_size,
      shuffle = True,
      seed = 123623
    )

    self.files = files
    self.files_dir = files_dir
    self.parse_output = parse_output
    self.create_input_tensor = create_input_tensor
    self.create_output_tensor = create_output_tensor

  def _get_batches_of_transformed_samples(self, indexes):
    inputs = []
    outputs = []

    for index in indexes:
      file = self.files[index]

      image = cv.imread(os.path.join(self.files_dir, file))
      image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

      output = self.parse_output(file)

      inputs.append(image)
      outputs.append(output)

    x = self.create_input_tensor(inputs)
    y = self.create_output_tensor(outputs)

    return x, y
