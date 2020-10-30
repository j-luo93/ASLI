from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

ext_modules = [Extension('sound_law.rl.reward',
                         ['sound_law/rl/reward.pyx'])]
setup(
    name='sound_law',
    version='0.1',
    ext_modules=cythonize(ext_modules, annotate=True),
    packages=find_packages())
