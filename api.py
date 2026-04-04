import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
import cv2
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

app = FastAPI(title="Pneumonia Detector API", version="2.0")

# Charger le modèle au démarrage
checkpoint = torch.load(
    "models/pneumonia_model.pth",
    map_location=torch.device("cpu")
)

model = models.resnet50(weights=None)
model.fc = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 2)
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

CLASSES   = checkpoint["classes"]
THRESHOLD = checkpoint["threshold"]

# Grad-CAM sur la dernière couche conv
cam = GradCAM(model=model, target_layers=[model.layer4[-1]])

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def image_to_base64(img_array):
    _, buffer = cv2.imencode(".png", img_array)
    return base64.b64encode(buffer).decode("utf-8")

@app.get("/")
def root():
    return {"message": "Pneumonia Detector API v2", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model": "ResNet50", "classes": CLASSES}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image    = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor   = transform(image).unsqueeze(0)

    # Inférence
    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0]

    normal_prob    = round(probs[0].item(), 4)
    pneumonia_prob = round(probs[1].item(), 4)
    prediction     = "PNEUMONIA" if pneumonia_prob >= THRESHOLD else "NORMAL"

    # Générer la Grad-CAM
    grayscale_cam = cam(input_tensor=tensor)[0]

    # Image originale normalisée pour overlay
    img_resized = np.array(image.resize((224, 224))) / 255.0
    img_float   = np.float32(img_resized)

    # Superposer la heatmap
    cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
    cam_bgr   = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    # Encoder en base64
    gradcam_b64 = image_to_base64(cam_bgr)

    return JSONResponse({
        "prediction"   : prediction,
        "confidence"   : round(max(normal_prob, pneumonia_prob) * 100, 2),
        "probabilities": {
            "NORMAL"   : normal_prob,
            "PNEUMONIA": pneumonia_prob,
        },
        "threshold"    : THRESHOLD,
        "file"         : file.filename,
        "gradcam_b64"  : gradcam_b64,
    })
