import os, unittest, tempfile, shutil
import cv2 as cv
import numpy as np

from jakso_ml.training_data import SampleImage
from jakso_ml.utils import write_json_to_file

class TestSampleImage(unittest.TestCase):
  def setUp(self):
    self.image_dir = os.path.join(tempfile.gettempdir(), 'sample_image_test')
    self.image_file_path = os.path.join(self.image_dir, '100_something.png')
    self.json_file_path = os.path.join(self.image_dir, '100_something.json')
    os.makedirs(self.image_dir, exist_ok = True)

    image = np.zeros((128, 256, 3), dtype = 'uint8')
    # Set some value to non-zero so that we can test the
    # correct image was read.
    image[5, 47, 2] = 123
    cv.imwrite(self.image_file_path, image)

    # The meta data json.
    meta_data = {
      'value': 100,
      'somethingCustom': 'hello',
      'roi': {
        'x': 0.1,
        'y': 0.1,
        'width': 0.25,
        'height': 0.25
      }
    }

    write_json_to_file(meta_data, self.json_file_path)

  def tearDown(self):
    shutil.rmtree(self.image_dir)

  def test_create_new_sample(self):
    sample = SampleImage()
    sample.file_dir = self.image_dir
    sample.file_extension = 'png'
    sample.file_name_parts = ['500', 'temp']
    # The file_name_parts array should be mutable.
    sample.file_name_parts[1] = 'new_sample'
    sample.image = np.zeros((128, 256, 3), dtype = 'uint8')
    sample.image[5, 47, 2] = 252
    sample.meta_data = {'value': 500}
    sample.save()

    sample = SampleImage.read_from_file(os.path.join(self.image_dir, '500_new_sample.png'))
    self.assertEqual(sample.image[5, 47, 2], 252)
    self.assertEqual(sample.value, 500)

  def test_path_properties(self):
    sample = SampleImage.read_from_file(self.image_file_path)

    self.assertEqual(sample.file_path, self.image_file_path)
    self.assertEqual(sample.file_name, '100_something')
    self.assertEqual(sample.file_extension, 'png')
    self.assertEqual(sample.file_name_parts, ['100', 'something'])

  def test_set_file_name_parts(self):
    sample = SampleImage.read_from_file(self.image_file_path)
    sample.file_name_parts = ['200', 'somethingElse']

    self.assertEqual(sample.file_name, '200_somethingElse')

  def test_meta_data(self):
    sample = SampleImage.read_from_file(self.image_file_path)

    self.assertEqual(sample.meta_data.value, sample.value)
    self.assertEqual(sample.value, 100)
    self.assertEqual(sample.roi, (26, 13, 64, 32))
    self.assertEqual(sample.meta_data['somethingCustom'], 'hello')
    self.assertEqual(sample.meta_data['doesntExist'], None)

  def test_image_methods(self):
    sample = SampleImage.read_from_file(self.image_file_path)

    self.assertEqual(sample.image.shape, (128, 256, 3))
    self.assertEqual(sample.image[5, 47, 2], 123)
    self.assertEqual(sample.image[0, 0, 0], 0)

  def test_save(self):
    sample = SampleImage.read_from_file(self.image_file_path)

    sample.image[3, 2, 1] = 123
    sample.value = 1000
    sample.meta_data['helloAddValue'] = [True, 'Stuff']
    sample.save()
    sample = SampleImage.read_from_file(self.image_file_path)

    self.assertEqual(sample.image.shape, (128, 256, 3))
    self.assertEqual(sample.image[3, 2, 1], 123)
    self.assertEqual(sample.image[0, 0, 0], 0)
    self.assertEqual(sample.value, 1000)
    self.assertEqual(sample.meta_data['helloAddValue'], [True, 'Stuff'])

  def test_change_file_path_before_accessing_image_or_meta_data(self):
    sample = SampleImage.read_from_file(self.image_file_path)
    sample.file_name_parts = ['200', 'different']

    self.assertEqual(sample.image[5, 47, 2], 123)
    self.assertEqual(sample.value, 100)

  def test_save_with_changed_file_path(self):
    sample = SampleImage.read_from_file(self.image_file_path)

    sample.image[3, 2, 1] = 123
    sample.value = 1000
    sample.meta_data['helloAddValue'] = [True, 'Stuff']
    sample.file_name_parts = ['200', 'different']
    sample.save()
    sample = SampleImage.read_from_file(os.path.join(self.image_dir, '200_different.png'))

    self.assertEqual(sample.image.shape, (128, 256, 3))
    self.assertEqual(sample.image[3, 2, 1], 123)
    self.assertEqual(sample.value, 1000)
    self.assertEqual(sample.meta_data['helloAddValue'], [True, 'Stuff'])

  def test_move(self):
    sample = SampleImage.read_from_file(self.image_file_path)

    sample.file_name_parts = ['300', 'moved']
    sample.move_files()
    sample = SampleImage.read_from_file(os.path.join(self.image_dir, '300_moved.png'))

    self.assertEqual(sample.meta_data.value, sample.value)
    self.assertEqual(sample.image[5, 47, 2], 123)
    self.assertEqual(os.path.isfile(self.image_file_path), False)

  def test_copy(self):
    sample = SampleImage.read_from_file(self.image_file_path)

    sample.file_name_parts = ['300', 'copied']
    sample.copy_files()

    sample1 = SampleImage.read_from_file(self.image_file_path)
    sample2 = SampleImage.read_from_file(os.path.join(self.image_dir, '300_copied.png'))

    for sample in [sample1, sample2]:
      self.assertEqual(sample.meta_data.value, sample.value)
      self.assertEqual(sample.image[5, 47, 2], 123)

  def test_read_all_from_dir(self):
    nested_dir1 = os.path.join(self.image_dir, 'nested_dir1')
    nested_dir2 = os.path.join(self.image_dir, 'nested_dir2')
    deeply_nested_dir = os.path.join(nested_dir2, 'deep_dir')

    os.makedirs(nested_dir1, exist_ok = True)
    os.makedirs(nested_dir2, exist_ok = True)
    os.makedirs(deeply_nested_dir, exist_ok = True)

    for i in [1, 2]:
      image = np.zeros((128, 256, 3), dtype = 'uint8')
      image[5, 47, 2] = i
      file_path = os.path.join(nested_dir1, f'img_{i}')
      cv.imwrite(file_path + '.png', image)
      write_json_to_file({'value': 10 * i}, file_path + '.json')

    for i in [3, 4]:
      image = np.zeros((128, 256, 3), dtype = 'uint8')
      image[5, 47, 2] = i
      file_path = os.path.join(nested_dir2, f'img_{i}')
      cv.imwrite(file_path + '.png', image)
      write_json_to_file({'value': 10 * i}, file_path + '.json')

    for i in [5, 6]:
      image = np.zeros((128, 256, 3), dtype = 'uint8')
      image[5, 47, 2] = i
      file_path = os.path.join(deeply_nested_dir, f'img_{i}')
      cv.imwrite(file_path + '.png', image)
      write_json_to_file({'value': 10 * i}, file_path + '.json')

    samples = SampleImage.read_all_from_dir(self.image_dir)
    img_values = sorted([sample.image[5, 47, 2] for sample in samples])
    values = sorted([sample.value for sample in samples])

    self.assertEqual(img_values, [1, 2, 3, 4, 5, 6, 123])
    self.assertEqual(values, [10, 20, 30, 40, 50, 60, 100])
