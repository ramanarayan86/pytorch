from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(name='batchnorm',
      #packages=['ptdebug'],
      py_modules = ['batchnorm_ext'], 
      ext_modules=[CppExtension('batchnorm_cpp', ['BatchnormExt_v3.cpp'], extra_compile_args=['-g','-funroll-loops' ,'-fopenmp', '-march=native' ])],
      cmdclass={'build_ext': BuildExtension})
