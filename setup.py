"""Builds the optional C++ re-check kernel. If compiling fails, the package
falls back to its numba implementation."""
import sys
from setuptools import setup, Extension

args = ["-O3", "-march=native", "-std=c++17"]
link = []
if sys.platform.startswith("linux"):
    args.append("-fopenmp")
    link.append("-fopenmp")

setup(ext_modules=[Extension("e8rerank._kernel", ["e8rerank/_kernel.cpp"], language="c++",
                             extra_compile_args=args, extra_link_args=link,
                             optional=True)])
