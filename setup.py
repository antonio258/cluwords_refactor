from setuptools import setup, find_packages
import os
import pathlib

here = pathlib.Path(__file__).parent.resolve()
requirements = f'{os.path.dirname(os.path.realpath(__file__))}/requirements.txt'

if os.path.isfile(requirements):
   with open(requirements) as f:
      install_requires = f.read().splitlines()


setup(
   name='cluwords_module',
   version='1.0',
   description='Cluwords implementation',
   author='Antonio Pereira',
   packages=find_packages(exclude='docs'),  #same as name
   install_requires=install_requires, #external packages as dependencies
   entry_points='''
        [console_scripts]
        cluwords=cluwords_module.buildClu:run_consule
   ''',
)