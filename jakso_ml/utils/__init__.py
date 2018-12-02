import json

def read_json_from_file(file_path):
  with open(file_path, encoding='utf-8') as data_file:
    return json.loads(data_file.read())

def round_tuple(tup):
  return tuple(map(lambda it: int(round(it)), tup))

def intersection(r1, r2):
  """
  Intersection of two rectangles
  """
  x1, y1, w1, h1 = r1
  x2, y2, w2, h2 = r2

  x = max(x1, x2)
  y = max(y1,y2)
  w = min(x1 + w1, x2 + w2) - x
  h = min(y1 + h1, y2 + h2) - y

  if w <= 0 or h <= 0:
    return (0, 0, 0, 0)

  return (x, y, w, h)

def area(r):
  """
  Area of a rectangle
  """
  return r[2] * r[3]
