from setuptools import setup, Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy as np

NAME = "hd-falcon-utils"
VERSION = "1.0"
DESCR = "HD-falcon's utils functions using Cython"
URL = "http://www.google.com"
REQUIRES = ['numpy', 'cython']

AUTHOR = "Weihong Xu"
EMAIL = "wexu@ucsd.edu"

LICENSE = "Apache 2.0"

SRC_DIR = "hd_falcon_utils"
PACKAGES = [SRC_DIR]

ext = Extension(
    SRC_DIR + ".wrapped",
    [SRC_DIR + "/wrapped.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=["-O3"], language="c++"
    )


EXTENSIONS = [ext]

if __name__ == "__main__":
    setup(install_requires=REQUIRES,
          packages=PACKAGES,
          zip_safe=False,
          name=NAME,
          version=VERSION,
          description=DESCR,
          author=AUTHOR,
          author_email=EMAIL,
          url=URL,
          license=LICENSE,
          cmdclass={"build_ext": build_ext},
          ext_modules=cythonize(EXTENSIONS)
          )