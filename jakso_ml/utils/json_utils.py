import json

def read_json_from_file(file_path):
  with open(file_path, encoding='utf-8') as data_file:
    return json.loads(data_file.read())
