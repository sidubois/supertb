from setuptools import setup

setup(name='supertb',
      version='1.0.1',
      description='A package to perform tight-binding calculations',
      url='https://github.com/sidubois/supertb',
      author='Simon M.-M. Dubois',
      author_email='smmdub@gmail.com',
      license='MIT',
      packages=['supertb'],
      install_requires=[
          'numpy',
          'scipy',
          'networkx',
          'pymatgen',
          'bottleneck', 
      ],
      zip_safe=False)
