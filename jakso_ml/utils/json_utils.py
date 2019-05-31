import json

def read_json_from_file(file_path):
  with open(file_path, encoding='utf-8') as file:
    return json.loads(file.read())

def write_json_to_file(dict, file_path):
  with open(file_path, 'w') as file:
    json.dump(dict, file, indent = 2)
