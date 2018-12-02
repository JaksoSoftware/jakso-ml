import os
import cv2 as cv
import numpy as np
from ..utils import read_json_from_file, round_tuple

__all__ = ['SampleMetaData', 'SampleImage']

class SampleMetaData(object):
  """
  The metadata saved as a JSON file with each input sample image.
  """
  def __init__(self, file_path, meta_dict):
    self.file_path = file_path
    self.dict = meta_dict

  @classmethod
  def read_from_file(cls, meta_file_path):
    meta_data_dict = read_json_from_file(meta_file_path)
    return cls(meta_file_path, meta_data_dict)

  @property
  def roi(self):
    roi = self.dict['roi']
    return (roi['x'], roi['y'], roi['width'], roi['height'])

  @property
  def value(self):
    return self.dict['value']

class SampleImage(object):
  SampleMetaData = SampleMetaData

  """
  A training sample (both input and augmented)
  """
  def __init__(self, file_path, image, meta_data):
    self.file_path = file_path
    self.image = image
    self.meta_data = meta_data
    self.roi = self.original_roi

  @classmethod
  def read_from_file(cls, image_file_path):
    meta_file_path = _remove_extension(image_file_path) + '.json'
    meta_data = None

    if os.path.exists(meta_file_path):
      meta_data = cls.SampleMetaData.read_from_file(meta_file_path)

    image = cv.imread(image_file_path)
    return cls(image_file_path, image, meta_data)

  def __copy__(self):
    copy = self.__class__(self.file_path, self.image, self.meta_data)
    copy.roi = self.roi
    return copy

  def __deepcopy__(self, _):
    clone = self.__copy__()
    clone.image = np.copy(clone.image)
    return clone

  @property
  def value(self):
    return self.meta_data.value

  @property
  def original_roi(self):
    ih, iw, _ = self.image.shape

    if self.meta_data != None:
      x, y, w, h = self.meta_data.roi
      return round_tuple((x * iw, y * ih, w * iw, h * ih))
    else:
      return (0, 0, iw, ih)

  @property
  def roi_center(self):
    x, y, w, h = self.roi
    return round_tuple((x + w / 2.0, y + h / 2.0))

  @property
  def roi_image(self):
    x, y, w, h = self.roi
    return self.image[y:(y + h), x:(x + w)]

  @property
  def file_name(self):
    return os.path.basename(self.file_path)

  @property
  def file_name_parts(self):
    return _remove_extension(self.file_name).split('_')

def _remove_extension(file_name):
  return '.'.join(file_name.split('.')[0:-1])
