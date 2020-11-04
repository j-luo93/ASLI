from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

ext_modules = [Extension('sound_law.rl.reward',
                         ['sound_law/rl/reward.pyx']),
               Extension('sound_law.rl.env_step',
                         ['sound_law/rl/env_step.pyx'])]
setup(
    name='sound_law',
    version='0.1',
    ext_modules=cythonize(ext_modules, annotate=True, language_level='3'),
    packages=find_packages())
