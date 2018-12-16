import copy

class Processor(object):
  '''
  Shared base class for all processors.

  A Processor is simply an object with a `process` method that takes in
  a SampleImage and returns an array of SampleImages.
  '''
  def process(self, sample):
    return [sample]

  def copy(self, **kwargs):
    props = copy.copy(self.__dict__)

    for key in kwargs:
      props[key] = kwargs[key]

    return self.__class__(**props)
