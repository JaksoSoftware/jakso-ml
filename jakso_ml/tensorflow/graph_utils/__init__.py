def find_sub_graphs(graph_def, filter):
  """
  Finds all sub graphs for the name of which `filter` return True.
  """
  graphs_by_name = {}

  for i in range(len(graph_def.node)):
    node = graph_def.node[i]

    if filter(node):
      graph_name = get_graph_name(node.name)

      if graph_name not in graphs_by_name:
        graphs_by_name[graph_name] = []

      graphs_by_name[graph_name].append(i)

  return graphs_by_name

def filter_sub_graph(graph_def, sub_graph, filter):
  new_sub_graph = []
  all_nodes_by_name = graph_to_dict(graph_def)

  for i in sub_graph:
    node = graph_def.node[i]

    if filter(node):
      new_sub_graph.append(i)

    inputs = node.input[:]
    del node.input[:]

    for input_name in inputs:
      input_node_name = get_node_name_for_input_name(input_name)

      if input_node_name not in all_nodes_by_name:
        node.input.append(input_name)
        continue

      input_node = all_nodes_by_name[input_node_name]

      if filter(input_node):
        node.input.append(input_name)

  return new_sub_graph

def find_sub_graph_inputs(graph_def, sub_graph):
  """
  Find nodes in the sub graph whose each input comes from outside the sub graph.
  """
  nodes_by_name = sub_graph_to_dict(graph_def, sub_graph)
  all_nodes_by_name = graph_to_dict(graph_def)
  input_nodes = []

  for i in sub_graph:
    node = graph_def.node[i]

    if len(node.input) == 0:
      continue

    inputs_not_in_sub_graph = []

    for input_name in node.input:
      input_node_name = get_node_name_for_input_name(input_name)

      if input_node_name not in nodes_by_name and input_node_name in all_nodes_by_name:
        inputs_not_in_sub_graph.append(all_nodes_by_name[input_node_name])

    if len(inputs_not_in_sub_graph) > 0:
      input_nodes.append((node, inputs_not_in_sub_graph))

  return input_nodes

def find_sub_graph_outputs(graph_def, sub_graph):
  """
  Find nodes in the sub graph whose each input comes from outside the sub graph.
  """
  nodes_by_name = sub_graph_to_dict(graph_def, sub_graph)
  output_nodes = []

  for i in sub_graph:
    node = graph_def.node[i]
    is_input_of_node_in_sub_graph = False

    for j in sub_graph:
      for input_name in graph_def.node[j].input:
        if get_node_name_for_input_name(input_name) == node.name:
          is_input_of_node_in_sub_graph = True
          break

      if is_input_of_node_in_sub_graph:
        break

    if not is_input_of_node_in_sub_graph:
      output_nodes.append(node)

  return output_nodes

def graph_to_dict(graph_def):
  return { node.name: node for node in graph_def.node }

def sub_graph_to_dict(graph_def, sub_graph):
  return { graph_def.node[i].name: graph_def.node[i] for i in sub_graph }

def get_graph_name(node_name):
  return node_name.split('/')[0]

def get_node_name_for_input_name(input_name):
  return input_name.split(':')[0]
