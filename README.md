# 🫁 Pneumonia Detector — MLOps End-to-End Project

> Système de détection automatique de pneumonie sur radios pulmonaires,
> construit avec ResNet50, FastAPI, MLflow et Streamlit.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![MLflow](https://img.shields.io/badge/MLflow-3.10-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.43-red)
![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.9983-brightgreen)

---

## 🎯 Objectif

Transformer une radio pulmonaire brute en diagnostic automatique
**Normal / Pneumonie** accessible via une interface web,
avec explicabilité visuelle (Grad-CAM) et traçabilité complète (MLflow).

---

## 🏗️ Architecture
---

## 📊 Résultats des expériences

| Modèle | AUC-ROC | F1 Macro | Recall Pneumonie | Faux Négatifs |
|--------|---------|----------|-----------------|---------------|
| CNN from scratch | 0.9838 | 0.00 ❌ | - | - |
| ResNet50 phase1 | 0.9838 | 0.00 ❌ | - | - |
| **ResNet50 phase2** | **0.9983** ✅ | **0.9768** ✅ | **0.982** ✅ | **14** |

> Expériences trackées avec **MLflow** et comparées via Parallel Coordinates Plot.

---

## ✨ Fonctionnalités

- 🔬 **Diagnostic automatique** Normal / Pneumonie
- 🌡️ **Grad-CAM** — visualisation des zones analysées par le modèle
- 📋 **Historique** des diagnostics avec horodatage
- 🚀 **API REST** FastAPI avec endpoint `/predict` et `/health`
- 📦 **MLflow** — tracking, versioning et Model Registry
- 🏆 **Model Registry** — alias `champion` pour le modèle en production

---

## 🛠️ Stack technique

| Composant | Technologie |
|-----------|-------------|
| Modèle | ResNet50 Transfer Learning (PyTorch) |
| Tracking | MLflow 3.10 |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Explicabilité | Grad-CAM (pytorch-grad-cam) |
| Dataset | Chest X-Ray Images (5263 images) |

---

## 🚀 Installation et lancement

### 1. Cloner le repo
```bash
git clone https://github.com/TON_USERNAME/pneumonia-api.git
cd pneumonia-api
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Lancer l'API FastAPI
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Lancer l'interface Streamlit
```bash
streamlit run app.py --server.port 8501
```

### 5. Ouvrir l'interface
---

## 📁 Structure du projet
---

## 🔬 Utilisation de l'API

### Health check
```bash
curl http://localhost:8000/health
```

### Diagnostic d'une radio
```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@radio.jpeg"
```

### Réponse
```json
{
  "prediction"    : "PNEUMONIA",
  "confidence"    : 100.0,
  "probabilities" : {
    "NORMAL"    : 0.0,
    "PNEUMONIA" : 1.0
  },
  "threshold"     : 0.5,
  "gradcam_b64"   : "..."
}
```

---

## 🧠 MLflow — Tracking & Registry
```bash
# Lancer l'interface MLflow
mlflow ui --port 5000 --backend-store-uri sqlite:///mlflow.db
# Ouvrir : http://localhost:5000
```

---

## ⚖️ Enjeux médicaux

> ⚠️ Ce projet est réalisé à des fins **éducatives**.
> Un faux négatif (pneumonie non détectée) peut avoir des conséquences graves.
> Ce système ne remplace pas le diagnostic d'un médecin qualifié.

---

## 👤 Auteur

**Kamagate** — Étudiant Master 2 Science des Données
