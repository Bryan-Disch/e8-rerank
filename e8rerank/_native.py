"""Loads the compiled C++ kernel (built at install time), if present."""
import ctypes
import glob
import os

P = ctypes.c_void_p
_i, _l, _f, _d = ctypes.c_int, ctypes.c_int64, ctypes.c_float, ctypes.c_double


def _load():
    here = os.path.dirname(__file__)
    for path in glob.glob(os.path.join(here, "_kernel*.so")) + glob.glob(os.path.join(here, "_kernel*.dylib")):
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            continue
        lib.e8_rerank.argtypes = [_i] * 6 + [P] * 7 + [_i, _i, P, P]
        lib.e8_rerank.restype = None
        lib.e8_encode.argtypes = [_l, P, P, P, P, _l, P, _d, _i, P]
        lib.e8_encode.restype = None
        lib.e8_neighbours.argtypes = [P, P, P, _l, P, _i, P]
        lib.e8_neighbours.restype = None
        lib.e8_score_aware.argtypes = [_l, _i, P, P, P, P, _f, P, P, _f, _i, _i]
        lib.e8_score_aware.restype = None
        return lib
    return None


KERNEL = _load()
THREADS = os.cpu_count() or 1
