from setuptools import setup, find_packages

setup(
  name = 'jakso-ml',
  version = '0.1',
  description = 'Machine learning utilities built by Jakso Software Oy',
  url = 'http://github.com/jakso/jakso_ml',
  author = 'Sami Koskimäki',
  author_email = 'sami@jakso.me',
  license = 'MIT',
  packages = find_packages(),
  zip_safe = False,
  install_requires = [
    'tensorflow',
    'opencv-python',
    'scipy'
  ]
)
