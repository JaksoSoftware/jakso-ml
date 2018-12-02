__all__ = ['combine_processors']

def combine_processors(processors):
  def process(samples):
    generated_samples = []

    for sample in samples:
      generated_samples.extend(augment_sample(sample))

    return generated_samples

  def augment_sample(sample):
    generated_samples = [sample]

    for processor in processors:
      generated_samples = run_processor(processor, generated_samples)

    return generated_samples

  def run_processor(processor, samples):
    generated_samples = []

    for sample in samples:
      generated_samples.extend(processor.process(sample))

    return generated_samples

  return process
