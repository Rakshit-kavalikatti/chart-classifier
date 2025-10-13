
import os
import warnings
import logging

# ==========================================================
# 🔇 Suppress TensorFlow logs and warnings *before* import
# These environment variables must be set before TF loads.
# ==========================================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'          # Suppress INFO/WARNING/ERROR messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'         # Disable oneDNN optimization messages
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'         # Suppress extra verbose C++ logs
# ==========================================================

# ==========================================================
# 🧹 Disable all general Python warnings and logging messages
# ==========================================================
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
# ==========================================================


import json
import sys
import shutil
import platform

import sys
import json
import torch
from PIL import Image

from transformers import AutoModelForImageClassification, ViTImageProcessor

MODEL_NAME = "ds4sd/DocumentFigureClassifier"
