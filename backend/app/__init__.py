import os
# Must be set before torch or scikit-learn are imported to prevent
# OpenBLAS / OpenMP memory-allocation crashes on Windows.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
