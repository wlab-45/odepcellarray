from setuptools import setup
from Cython.Build import cythonize
import numpy
import glob

setup(
    ext_modules=cythonize(glob.glob("*.pyx"), language_level=3),
    include_dirs=[numpy.get_include()]
)
