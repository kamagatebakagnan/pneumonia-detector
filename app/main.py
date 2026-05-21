# ============================================================
# API FastAPI — Détection de pneumonie
# ============================================================
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import io

# ── Initialisation de l'app ──────────────────────────────────
app = FastAPI(
    title="Pneumonia Detection API",
    description="API de détection de pneumonie sur radiographies pulmonaires",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Chargement du modèle ─────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/pneumonia_model.pth"

def load_model():
    """Charge ResNet50 avec les poids sauvegardés."""
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, 2)
    )
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    return model, checkpoint.get('threshold', 0.5)

try:
    model, threshold = load_model()
    print(f"Modèle chargé sur {device}")
    print(f"Seuil de décision : {threshold}")
except Exception as e:
    print(f"Erreur chargement modèle : {e}")
    model, threshold = None, 0.5

# ── Preprocessing ────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ── Endpoints ────────────────────────────────────────────────
@app.get("/")
def root():
    """Endpoint de santé — vérifie que l'API fonctionne."""
    return {
        "message"  : "Pneumonia Detection API",
        "status"   : "online",
        "model"    : "ResNet50",
        "device"   : str(device),
        "threshold": threshold
    }

@app.get("/health")
def health():
    """Vérifie l'état du modèle."""
    return {
        "status"      : "healthy" if model is not None else "error",
        "model_loaded": model is not None
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Prédit si une radiographie montre une pneumonie.

    - **file** : image radiographie (JPG, PNG)
    - **returns** : diagnostic, probabilité, recommandation
    """
    if model is None:
        raise HTTPException(status_code=503,
                           detail="Modèle non disponible")

    # Vérifier le type de fichier
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400,
                           detail="Le fichier doit être une image")

    try:
        # Lire et préprocesser l'image
        contents = await file.read()
        image    = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor   = transform(image).unsqueeze(0).to(device)

        # Prédiction
        with torch.no_grad():
            outputs = model(tensor)
            probs   = torch.softmax(outputs, dim=1)[0]
            prob_pneumonia = probs[1].item()
            prob_normal    = probs[0].item()

        # Décision selon le seuil
        diagnostic = "PNEUMONIA" if prob_pneumonia >= threshold else "NORMAL"

        # Recommandation médicale
        if diagnostic == "PNEUMONIA":
            if prob_pneumonia >= 0.90:
                recommandation = "Forte suspicion de pneumonie — consultation urgente recommandée"
            else:
                recommandation = "Suspicion de pneumonie — consultation médicale recommandée"
        else:
            if prob_normal >= 0.90:
                recommandation = "Radiographie normale — pas de signe de pneumonie détecté"
            else:
                recommandation = "Radiographie probablement normale — suivi recommandé"

        return {
            "diagnostic"        : diagnostic,
            "probabilite_pneumonie": round(prob_pneumonia, 4),
            "probabilite_normal"   : round(prob_normal, 4),
            "seuil_utilise"     : threshold,
            "confiance"         : round(max(prob_pneumonia, prob_normal), 4),
            "recommandation"    : recommandation,
            "fichier_analyse"   : file.filename
        }

    except Exception as e:
        raise HTTPException(status_code=500,
                           detail=f"Erreur lors de l'analyse : {str(e)}")
