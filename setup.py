from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup


def get_ext(*args, **kwargs):
    return Extension(*args,
                     extra_link_args=['-L/usr/lib/x86_64-linux-gnu/', '-std=c++11', '-fopenmp'],
                     extra_compile_args=['-std=c++11', '-fopenmp'],
                     undef_macros=['NDEBUG'],
                     **kwargs)


ext_modules = [get_ext('sound_law.rl.reward', ['sound_law/rl/reward.pyx']),
               get_ext('sound_law.rl.mcts_fast', ['sound_law/rl/mcts_fast.pyx'])]

setup(
    name='sound_law',
    version='0.1',
    ext_modules=cythonize(ext_modules, language_level='3'),
    packages=find_packages())
