from setuptools import setup, find_packages

setup(
  name = 'jakso-ml',
  version = '0.3',
  description = 'Machine learning utilities built by Jakso Software Oy',
  url = 'https://github.com/JaksoSoftware/jakso-ml',
  author = 'Sami Koskimäki',
  author_email = 'sami@jakso.me',
  license = 'MIT',
  packages = find_packages(exclude = ['tests']),
  zip_safe = False,
  install_requires = [
    'tensorflow-gpu',
    'opencv-python',
    'scipy'
  ]
)
