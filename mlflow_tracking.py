# ============================================================
# MLFLOW — Tracking des expériences
# ============================================================
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torchvision import models
import numpy as np

# Démarrer MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("pneumonia-detection")

# Simuler les résultats de nos 2 modèles
experiments = [
    {
        "name"      : "CNN_from_scratch",
        "params"    : {
            "architecture" : "CNN_scratch",
            "epochs"       : 9,
            "learning_rate": 0.001,
            "batch_size"   : 32,
            "optimizer"    : "Adam",
            "augmentation" : True,
        },
        "metrics"   : {
            "auc_roc"          : 0.9838,
            "f1_macro"         : 0.00,
            "recall_pneumonia" : 0.915,
            "recall_normal"    : 0.624,
            "false_negatives"  : 33,
            "accuracy"         : 0.81,
        }
    },
    {
        "name"      : "ResNet50_phase1",
        "params"    : {
            "architecture" : "ResNet50_transfer",
            "epochs"       : 5,
            "learning_rate": 0.001,
            "batch_size"   : 32,
            "optimizer"    : "Adam",
            "frozen_layers": "all_except_fc",
            "pretrained"   : "ImageNet",
        },
        "metrics"   : {
            "auc_roc"          : 0.9838,
            "f1_macro"         : 0.0,
            "recall_pneumonia" : 0.0,
            "false_negatives"  : 0,
            "accuracy"         : 0.0,
        }
    },
    {
        "name"      : "ResNet50_phase2",
        "params"    : {
            "architecture" : "ResNet50_transfer",
            "epochs"       : 10,
            "learning_rate": 0.0001,
            "batch_size"   : 32,
            "optimizer"    : "Adam",
            "frozen_layers": "layer1_to_3",
            "pretrained"   : "ImageNet",
            "unfrozen"     : "layer4_fc",
        },
        "metrics"   : {
            "auc_roc"          : 0.9983,
            "f1_macro"         : 0.9768,
            "recall_pneumonia" : 0.982,
            "precision_normal" : 0.95,
            "false_negatives"  : 14,
            "accuracy"         : 0.98,
        }
    }
]

# Logger chaque expérience dans MLflow
print("Logging des expériences dans MLflow...\n")

for exp in experiments:
    with mlflow.start_run(run_name=exp["name"]):
        # Logger les hyperparamètres
        mlflow.log_params(exp["params"])

        # Logger les métriques
        mlflow.log_metrics(exp["metrics"])

        # Tags utiles
        mlflow.set_tag("dataset",  "Chest X-Ray (5263 images)")
        mlflow.set_tag("task",     "binary_classification")
        mlflow.set_tag("domain",   "medical_imaging")
        mlflow.set_tag("model",    exp["name"])

        print(f"  Run '{exp['name']}' loggé :")
        print(f"    AUC-ROC          : {exp['metrics']['auc_roc']}")
        print(f"    Faux Négatifs    : {exp['metrics']['false_negatives']}")
        print()

print("Tous les runs sont enregistrés !")
print("Lance : mlflow ui --port 5000")
print("Puis ouvre : http://localhost:5000")
