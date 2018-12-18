import os
import tensorflow as tf

from tensorflow import keras
from .export_keras_model_for_mobile import export_keras_model_for_mobile

class KerasMobileModelSaveCallback(keras.callbacks.Callback):
  '''
  A keras callback that periodically saves the graph as a .pb file that
  can be used with tensorflow mobile.
  '''
  def __init__(self, model, file_path, batches_between_saves):
    super().__init__()

    self.model = model
    self.file_path = file_path
    self.batches_between_saves = batches_between_saves

  def on_batch_end(self, batch, logs):
    if batch % self.batches_between_saves == 0:
      graph_def = export_keras_model_for_mobile(self.model)
      file_path = self.file_path() if callable(self.file_path) else self.file_path

      tf.train.write_graph(
        graph_def,
        os.path.dirname(file_path),
        os.path.basename(file_path),
        as_text = False
      )

      print(
        '\n',
        'graph saved!',
        'input:', graph_def.node[0].name,
        'output:', graph_def.node[-1].name
      )
