import tensorflow as tf

from tensorflow import keras
from .export_keras_model_for_mobile import export_keras_model_for_mobile

class KerasMobileModelSaveCallback(keras.callbacks.Callback):
  '''
  A keras callback that periodically saves the graph as a .pb file that
  can be used with tensorflow mobile.
  '''
  def __init__(self, model, file_dir, file_name, batches_between_saves):
    super().__init__()

    self.model = model
    self.file_dir = file_dir
    self.file_name = file_name
    self.batches_between_saves = batches_between_saves

  def on_batch_end(self, batch, logs):
    if batch % self.batches_between_saves == 0:
      graph_def = export_keras_model_for_mobile(self.model)

      tf.train.write_graph(
        graph_def,
        self.file_dir,
        self.file_name,
        as_text = False
      )

      print(
        '\n',
        'graph saved!',
        'input:', graph_def.node[0].name,
        'output:', graph_def.node[-1].name
      )
