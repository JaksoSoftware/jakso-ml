def intersection(r1, r2):
  '''
  Intersection of two rectangles
  '''
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
  '''
  Area of a rectangle
  '''
  return r[2] * r[3]
