import os

from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup


def get_ext(*args, **kwargs):
    extra_args = ['-L/usr/lib/x86_64-linux-gnu/', '-std=c++17', '-fopenmp', '-lpthread', '-lboost_thread',
                  '-DBOOST_ERROR_CODE_HEADER_ONLY', '-lboost_system', '-O3', '-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_ERROR']
    return Extension(*args,
                     extra_link_args=extra_args,
                     extra_compile_args=extra_args,
                     undef_macros=['NDEBUG'],
                     **kwargs)


os.environ["CC"] = "g++"
ext_modules = [get_ext('sound_law.rl.reward', ['sound_law/rl/reward.pyx']),
               get_ext('sound_law.rl.mcts_cpp.mcts_cpp', ['sound_law/rl/mcts_cpp/mcts_cpp.pyx'])]

setup(
    name='sound_law',
    version='0.1',
    ext_modules=cythonize(ext_modules, language_level='3'),
    packages=find_packages())
