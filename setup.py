"""
pyprf setup.

"""
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))


with open('README.md', encoding='utf-8') as f:
    long_description = f.read()



setup(name='prfsim',
      version='0.1.10',
      description=('A free & open source python tool for population receptive \
                    field simulation of fMRI data.'),
      long_description=long_description,
      long_description_content_type='text/markdown',

      url='https://github.com/arash-ash/prfsim',
      download_url='https://github.com/arash-ash/prfsim/archive/v0.1.10.tar.gz',
      author='Arash Ash',
      author_email='arash.ashrafnejad@gmail.com',
      license='GNU General Public License Version 3',
      install_requires=['numpy', 'scipy', 'nibabel', 'pandas', 'seaborn'],
      keywords=['pRF', 'fMRI', 'retinotopy', 'simulation'],

#      packages=find_packages(exclude=['contrib', 'docs', 'tests']),
      py_modules=["sim"],
#      entry_points={
#         'console_scripts': [
#              'prfsim = prfsim.__main__:main',
#              ]},
      )

