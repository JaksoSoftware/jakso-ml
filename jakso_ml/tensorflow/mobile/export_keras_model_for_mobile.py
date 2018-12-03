import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.graph_util import convert_variables_to_constants, remove_training_nodes
from tensorflow.python.tools import optimize_for_inference_lib

from ..graph_utils import (
  get_node_name_for_input_name,
  find_sub_graphs,
  find_sub_graph_inputs,
  find_sub_graph_outputs
)

def export_keras_model_for_mobile(model):
  '''
  Given a Keras model, applies a bunch of conversions and optimizations for the
  graph to make it work on tensorflow mobile. Returns the exported tensorflow
  GraphDef.
  '''
  output_names = [out.op.name for out in model.outputs]
  graph = K.get_session().graph
  graph_def = None

  with graph.as_default():
    graph_def = graph.as_graph_def()
    graph_def = remove_training_nodes(graph_def)
    graph_def = clear_devices(graph_def)
    graph_def = freeze_graph(graph_def, output_names)
    graph_def = fix_batch_normalization(graph_def)
    graph_def = remove_dropout(graph_def)
    graph_def = remove_keras_training_nodes(graph_def)

    return graph_def

def clear_devices(graph_def):
  for node in graph_def.node:
    node.device = ''

  return graph_def

def freeze_graph(graph_def, output_names = None):
  '''
  Freezes the state of a graph into a pruned computation graph.

  Creates a new computation graph where variable nodes are replaced by
  constants taking their current value in the session. The new graph will be
  pruned so subgraphs that are not necessary to compute the requested
  outputs are removed.
  '''
  freeze_var_names = [v.op.name for v in tf.global_variables()]
  output_names += [v.op.name for v in tf.global_variables()]

  return convert_variables_to_constants(
    K.get_session(),
    graph_def,
    output_names,
    freeze_var_names
  )

def fix_batch_normalization(graph_def):
  '''
  Keras batch normalization doesn't work on tensorflow mobile by default.
  We need to simplify it by removing all nodes but the ones needed
  for inference.
  '''
  NODE_NAMES = ['gamma', 'beta', 'moving_mean', 'moving_variance', 'FusedBatchNorm_1']
  GAMMA, BETA, MEAN, VAR, BATCH_NORM = NODE_NAMES

  batch_norm_graphs = find_sub_graphs(
    graph_def,
    lambda node: node.name.startswith('batch_normalization')
  )

  nodes_to_remove = []

  for _, batch_norm_graph in batch_norm_graphs.items():
    nodes = {}

    for i in batch_norm_graph:
      node = graph_def.node[i]
      last_name_part = node.name.split('/')[-1]

      if last_name_part in NODE_NAMES:
        nodes[last_name_part] = node
      else:
        nodes_to_remove.append(node)

    inputting_node = find_sub_graph_inputting_node(graph_def, batch_norm_graph)
    output_node = find_sub_graph_output(graph_def, batch_norm_graph)

    nodes[BATCH_NORM].input[0] = inputting_node.name
    nodes[BATCH_NORM].input[1] = nodes[GAMMA].name
    nodes[BATCH_NORM].input[2] = nodes[BETA].name
    nodes[BATCH_NORM].input[3] = nodes[MEAN].name
    nodes[BATCH_NORM].input[4] = nodes[VAR].name

    graph_def = replace_input(
      graph_def,
      output_node.name,
      nodes[BATCH_NORM].name
    )

  for node in nodes_to_remove:
    graph_def.node.remove(node)

  return graph_def

def remove_dropout(graph_def):
  '''
  Remove Keras dropoout operations from the graph as they are only
  needed for training.
  '''
  dropout_graphs = find_sub_graphs(
    graph_def,
    lambda node: node.name.startswith('dropout')
  )

  nodes_to_remove = []

  for _, dropout_graph in dropout_graphs.items():
    inputting_node = find_sub_graph_inputting_node(graph_def, dropout_graph)
    output_node = find_sub_graph_output(graph_def, dropout_graph)

    # Connect the dropout input directly to the dropout output.
    graph_def = replace_input(
      graph_def,
      output_node.name,
      inputting_node.name
    )

    for i in dropout_graph:
      nodes_to_remove.append(graph_def.node[i])

  for node in nodes_to_remove:
    graph_def.node.remove(node)

  return graph_def

def remove_keras_training_nodes(graph_def):
  nodes_to_remove = []

  for node in graph_def.node:
    if node.name.startswith('training') or node.name.startswith('Adam'):
      nodes_to_remove.append(node)

  for node in nodes_to_remove:
    graph_def.node.remove(node)

  return graph_def

def find_sub_graph_inputting_node(graph_def, sub_graph):
  for _, inputting_nodes in find_sub_graph_inputs(graph_def, sub_graph):
    for inputting_node in inputting_nodes:
      if not is_keras_learning_phase_node(inputting_node):
        return inputting_node

  return None

def is_keras_learning_phase_node(node):
  return 'keras_learning_phase' in node.name

def find_sub_graph_output(graph_def, sub_graph):
  output_nodes = find_sub_graph_outputs(graph_def, sub_graph)
  return output_nodes[0]

def replace_input(graph_def, input_node_name, new_input_name):
  for node in graph_def.node:
    for i in range(len(node.input)):
      if get_node_name_for_input_name(node.input[i]) == input_node_name:
        node.input[i] = new_input_name

  return graph_def
