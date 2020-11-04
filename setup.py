from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup


def get_ext(*args, **kwargs):
    return Extension(*args,
                     extra_link_args=['-L/usr/lib/x86_64-linux-gnu/'],
                     **kwargs)


ext_modules = [get_ext('sound_law.rl.reward', ['sound_law/rl/reward.pyx']),
               get_ext('sound_law.rl.env_step', ['sound_law/rl/env_step.pyx'])]

setup(
    name='sound_law',
    version='0.1',
    ext_modules=cythonize(ext_modules, annotate=True, language_level='3'),
    packages=find_packages())
