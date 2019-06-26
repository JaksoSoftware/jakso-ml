import os, shutil, copy, datetime
import cv2 as cv
import numpy as np
from ..utils import read_json_from_file, write_json_to_file, round_tuple

class SampleMetaData(object):
  '''
  The metadata saved as a JSON file with each input sample image.
  '''
  def __init__(self, file_path = ''):
    self.__current_file_path = SampleImagePath(file_path)
    self.__file_path = SampleImagePath(file_path)
    self.__dict = None

  @classmethod
  def read_from_file(cls, file_path):
    return cls(file_path)

  @classmethod
  def createTimestamp(cls):
    iso_date = datetime.datetime.now().isoformat()
    return None

  @classmethod
  def parseTimestamp(cls, timstamp):
    return None

  @property
  def file_path(self):
    return str(self.__file_path)

  @file_path.setter
  def file_path(self, file_path):
    self.__file_path = SampleImagePath(file_path)

  @property
  def file_name(self):
    return self.__file_path.file_name

  @file_name.setter
  def file_name(self, file_name):
    self.__file_path.file_name = file_name

  @property
  def file_extension(self):
    return self.__file_path.extension

  @file_extension.setter
  def file_extension(self, file_extension):
    self.__file_path.extension = file_extension

  @property
  def file_dir(self):
    return self.__file_path.dir_name

  @file_dir.setter
  def file_dir(self, file_dir):
    self.__file_path.dir_name = file_dir

  @property
  def file_name_parts(self):
    return self.__file_path.name_parts

  @file_name_parts.setter
  def file_name_parts(self, file_name_parts):
    self.__file_path.name_parts = file_name_parts

  @property
  def dict(self):
    self.__ensure_dictionary()
    return self.__dict

  @dict.setter
  def dict(self, dict):
    self.__dict = dict

  @property
  def roi(self):
    roi = self['roi']
    return (roi['x'], roi['y'], roi['width'], roi['height'])

  @property
  def value(self):
    return self['value']

  @value.setter
  def value(self, value):
    self['value'] = value

  def save(self):
    if self.__dict is not None:
      write_json_to_file(self.__dict, self.file_path)

  def move_file(self):
    shutil.move(str(self.__current_file_path), str(self.file_path))
    self.__current_file_path = copy.copy(self.file_path)

  def copy_file(self):
    shutil.copy(str(self.__current_file_path), str(self.file_path))
    self.__current_file_path = copy.copy(self.file_path)

  def __getitem__(self, key):
    return self.dict[key] if key in self.dict else None

  def __setitem__(self, key, value):
    self.dict[key] = value

  def __ensure_dictionary(self):
    if self.__dict is None:
      file_path = str(self.__current_file_path)
      self.__dict = read_json_from_file(file_path) if os.path.isfile(file_path) else {}

class SampleImage(object):
  '''
  A training sample (both input and augmented)
  '''
  SampleMetaData = SampleMetaData

  def __init__(self, file_path = ''):
    self.__current_file_path = SampleImagePath(file_path)
    self.__file_path = SampleImagePath(file_path)

    self.__image = None
    self.__meta_data = None
    self.__roi = None

  @classmethod
  def read_from_file(cls, image_file_path):
    return cls(image_file_path)

  @classmethod
  def read_all_from_dir(cls, dir_path):
    files = os.listdir(dir_path)
    sample_images = []

    for file in files:
      file_path = os.path.join(dir_path, file)

      if os.path.isdir(file_path):
        sample_images += cls.read_all_from_dir(file_path)
        continue

      if _is_image_file(file):
        sample_images.append(cls.read_from_file(file_path))

    return sample_images

  def __copy__(self):
    copy = self.__class__(self.file_path)

    copy.__image = self.__image
    copy.__meta_data = self.__meta_data
    copy.__roi = self.__roi

    return copy

  def __deepcopy__(self, _):
    clone = self.__copy__()

    if clone.__image is not None:
      clone.__image = np.copy(clone.__image)

    return clone

  def move_files(self):
    shutil.move(str(self.__current_file_path), str(self.file_path))
    self.__current_file_path = copy.copy(self.file_path)

    if self.meta_data is not None:
      self.meta_data.move_file()

  def copy_files(self):
    shutil.copy(str(self.__current_file_path), str(self.file_path))
    self.__current_file_path = copy.copy(self.file_path)

    if self.meta_data is not None:
      self.meta_data.copy_file()

  def save(self):
    cv.imwrite(str(self.file_path), self.image)

    if self.meta_data is not None:
      self.meta_data.save()

  @property
  def image(self):
    if self.__image is None:
      # Load the image lazily.
      self.__image = cv.imread(str(self.__current_file_path))

    return self.__image

  @image.setter
  def image(self, image):
    self.__image = image

  @property
  def meta_data(self):
    if self.__meta_data is None:
      current_file_path = self.__current_file_path.get_path_with_extension('json')
      file_path = self.__file_path.get_path_with_extension('json')

      # Load meta data lazily.
      self.__meta_data = self.__class__.SampleMetaData.read_from_file(current_file_path)
      self.__meta_data.file_path = current_file_path

    return self.__meta_data

  @meta_data.setter
  def meta_data(self, meta_data):
    self.__meta_data = SampleMetaData(self.__file_path.get_path_with_extension('json'))
    self.__meta_data.dict = meta_data

  @property
  def value(self):
    return self.meta_data.value

  @value.setter
  def value(self, value):
    self.meta_data.value = value

  @property
  def roi(self):
    if self.__roi is None:
      self.__roi = self.original_roi

    return self.__roi

  @roi.setter
  def roi(self, roi):
    self.__roi = roi

  @property
  def original_roi(self):
    ih, iw, _ = self.image.shape

    if self.meta_data is not None:
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
  def file_path(self):
    return str(self.__file_path)

  @file_path.setter
  def file_path(self, file_path):
    self.__file_path = SampleImagePath(file_path)

    if self.meta_data is not None:
      self.meta_data.file_path = self.__file_path.get_path_with_extension('json')

  @property
  def file_name(self):
    return self.__file_path.file_name

  @file_name.setter
  def file_name(self, file_name):
    self.__file_path.file_name = file_name

    if self.meta_data is not None:
      self.meta_data.file_name = file_name

  @property
  def file_extension(self):
    return self.__file_path.extension

  @file_extension.setter
  def file_extension(self, file_extension):
    self.__file_path.extension = file_extension

    if self.meta_data is not None:
      self.meta_data.file_extension = file_extension

  @property
  def file_dir(self):
    return self.__file_path.dir_name

  @file_dir.setter
  def file_dir(self, file_dir):
    self.__file_path.dir_name = file_dir

    if self.meta_data is not None:
      self.meta_data.file_dir = file_dir

  @property
  def file_name_parts(self):
    return self.__file_path.name_parts

  @file_name_parts.setter
  def file_name_parts(self, file_name_parts):
    self.__file_path.name_parts = file_name_parts

    if self.meta_data is not None:
      self.meta_data.file_name_parts = file_name_parts

class SampleImagePath(object):
  def __init__(self, file_path):
    file_name = os.path.basename(file_path)
    self.__dir_name = os.path.dirname(file_path)
    self.__name_parts = self.__split_file_name(file_name)
    self.__extension = file_name.split('.')[-1]

  @property
  def file_name(self):
    return '_'.join(self.name_parts)

  @file_name.setter
  def file_name(self, file_name):
    self.name_parts = self.__split_file_name(file_name)

  @property
  def name_parts(self):
    return self.__name_parts

  @name_parts.setter
  def name_parts(self, name_parts):
    self.__name_parts = name_parts

  @property
  def extension(self):
    return self.__extension

  @extension.setter
  def extension(self, extension):
    self.__extension = extension

  @property
  def dir_name(self):
    return self.__dir_name

  @dir_name.setter
  def dir_name(self, dir_name):
    self.__dir_name = dir_name

  def get_path_with_extension(self, extension):
   return os.path.join(self.dir_name, self.file_name + '.' +  extension)

  def __split_file_name(self, file_name):
    return '.'.join(file_name.split('.')[0:-1]).split('_')

  def __str__(self):
    return self.get_path_with_extension(self.extension)

  def __copy__(self):
    return SampleImagePath(str(self))

def _remove_extension(file_name):
  return '.'.join(file_name.split('.')[0:-1])

def _is_image_file(file_name):
  file_name = file_name.lower()

  return (
    file_name.endswith('.jpg') or
    file_name.endswith('.jpeg') or
    file_name.endswith('.png')
  )
