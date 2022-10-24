from distutils.core import setup
from distutils.extension import Extension

import numpy
from Cython.Build import cythonize

# extensions = [
#     Extension("primes", ["primes.pyx"],
#         include_dirs=[...],
#         libraries=[...],
#         library_dirs=[...]),
#     # Everything but primes.pyx is included here.
#     Extension("*", ["*.pyx"],
#         include_dirs=[...],
#         libraries=[...],
#         library_dirs=[...]),
# ]

extensions = [
    Extension(
        "cython_metric", ["cython_metric.pyx"], include_dirs=[numpy.get_include()]
    ),
]

setup(
    name="cython_metric",
    ext_modules=cythonize(["cython_metric.pyx"], annotate=True, language_level="3"),
)
