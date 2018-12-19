import os
import tensorflow as tf

from tensorflow import keras
from .export_keras_model_for_mobile import export_keras_model_for_mobile

class KerasMobileModelSaveCallback(keras.callbacks.Callback):
  '''
  A keras callback that periodically saves the graph as a .pb file that
  can be used with tensorflow mobile.
  '''
  def __init__(self, model, file_path):
    super().__init__()

    self.model = model
    self.file_path = file_path

  def on_epoch_end(self, epoch, logs):
    graph_def = export_keras_model_for_mobile(self.model)
    val_loss = logs.get('val_loss')

    # `epoch + 1` because `epoch` is zero-based, while the epochs printed to the
    # console are one-based.
    file_path = self.file_path(epoch + 1, val_loss) if callable(self.file_path) else self.file_path

    if file_path != None:
      # file_path == None means that we don't want to save the model this
      # time around.
      tf.train.write_graph(
        graph_def,
        os.path.dirname(file_path),
        os.path.basename(file_path),
        as_text = False
      )

      print(
        '\n',
        'graph saved!',
        'val_loss:', val_loss,
        'input:', graph_def.node[0].name,
        'output:', graph_def.node[-1].name
      )
