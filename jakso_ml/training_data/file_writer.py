import os
import cv2 as cv
from .processor import Processor

__all__ = ['FileWriter']

class FileWriter(Processor):
  """
  Writes a sample to a file.
  """
  def __init__(self, folder_path, file_name_creator):
    self.folder_path = folder_path
    self.file_name_creator = file_name_creator

  def process(self, sample):
    if not os.path.exists(self.folder_path):
      os.makedirs(self.folder_path)

    file_path = os.path.join(self.folder_path, self.file_name_creator(sample))
    cv.imwrite(file_path, sample.image)

    return [sample]
