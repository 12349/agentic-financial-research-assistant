"""
experiments/__init__.py
"""
import os

# Prevent macOS libomp conflict between FAISS and PyTorch
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
