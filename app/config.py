# ============================================================
# Configuration de l'API
# ============================================================
import os

# Modèle
MODEL_PATH     = os.getenv("MODEL_PATH", "models/pneumonia_model.pth")
THRESHOLD      = float(os.getenv("THRESHOLD", "0.5"))
DEVICE         = os.getenv("DEVICE", "cpu")

# API
API_TITLE      = "Pneumonia Detection API"
API_VERSION    = "1.0.0"
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB max

# Classes
CLASSES = ["NORMAL", "PNEUMONIA"]
