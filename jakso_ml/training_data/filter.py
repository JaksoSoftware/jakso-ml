from .processor import Processor

__all__ = ['Filter']

class Filter(Processor):
  """
  Processor that only lets through SampleImages for which the
  filter function returns true.
  """
  def __init__(self, filter):
    self.filter = filter

  def process(self, sample):
    if self.filter(sample):
      return [sample]
    else:
      return []
