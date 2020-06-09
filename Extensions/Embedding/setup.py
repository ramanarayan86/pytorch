from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(name='embedding',
      #packages=['ptdebug'],
      py_modules = ['embedding_ext'], 
      ext_modules=[CppExtension('embedding_cpp', ['EmbeddingExt_v3.cpp'], extra_compile_args=['-g','-funroll-loops' ,'-fopenmp', '-march=native' ])],
      cmdclass={'build_ext': BuildExtension})
