"""Shared ctypes loader for the prebuilt libr4d library (r4d.so).

Loads r4d.so from site-packages and exposes the C entry points via ctypes.
Our r4d.so is located at /opt/vllm/lib/python3.12/site-packages/r4d.so.
"""

import ctypes
import functools
import os
import sys

_LIB_PATH = os.environ.get("R4D_LIB", "/opt/vllm/lib/python3.12/site-packages/r4d.so")


def available() -> bool:
    return os.path.exists(_LIB_PATH)


@functools.lru_cache
def import_r4d():
    """Import r4d.so as the pybind11 module, or None if unavailable."""
    if not available():
        return None
    libdir = os.path.dirname(_LIB_PATH)
    if libdir not in sys.path:
        sys.path.append(libdir)
    dlflags = sys.getdlopenflags()
    sys.setdlopenflags(os.RTLD_NOW | os.RTLD_DEEPBIND)
    try:
        import r4d
        return r4d
    except ImportError:
        return None
    finally:
        sys.setdlopenflags(dlflags)


@functools.lru_cache
def lib() -> ctypes.CDLL:
    lib = ctypes.CDLL(_LIB_PATH)
    # W4A16 skinny GEMM: gemm_w4a16_nt_m64(a, wq, wsz, c, M, K, N, WV, SK, MB, NPW, NT, stream)
    fn = getattr(lib, "r4d_gemm_w4a16_nt_m64", None)
    if fn is not None:
        fn.restype = None
        fn.argtypes = [ctypes.c_long] * 4 + [ctypes.c_int] * 8 + [ctypes.c_long]
    # Query functions
    for name in ("r4d_gemm_w4a16_nt_m64_max_m", "r4d_gemm_w4a16_nt_m64_group"):
        fn = getattr(lib, name, None)
        if fn is not None:
            fn.restype = ctypes.c_int
            fn.argtypes = []
    return lib


def has_w4a16() -> bool:
    """True when the loaded r4d.so carries the W4A16 skinny GEMM."""
    return available() and hasattr(lib(), "r4d_gemm_w4a16_nt_m64")
