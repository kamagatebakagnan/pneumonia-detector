import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torchvision.models as models

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("pneumonia-detection")

# Charger le checkpoint
checkpoint = torch.load(
    "models/pneumonia_model.pth",
    map_location=torch.device("cpu")
)

print("Classes     :", checkpoint["classes"])
print("Architecture:", checkpoint["architecture"])
print("Threshold   :", checkpoint["threshold"])

# Reconstruire la VRAIE architecture
model = models.resnet50(weights=None)
model.fc = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 2)
)

# Charger les poids
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("✅ Modèle chargé avec succès !")

# Logger dans MLflow
with mlflow.start_run(run_name="ResNet50_production"):
    mlflow.log_params({
        "architecture"  : checkpoint["architecture"],
        "epochs"        : 10,
        "learning_rate" : 0.0001,
        "batch_size"    : 32,
        "optimizer"     : "Adam",
        "classes"       : str(checkpoint["classes"]),
        "threshold"     : checkpoint["threshold"],
    })
    mlflow.log_metrics({
        "auc_roc"          : 0.9983,
        "f1_macro"         : 0.9768,
        "recall_pneumonia" : 0.982,
        "accuracy"         : 0.98,
        "false_negatives"  : 14,
    })
    mlflow.set_tags({
        "dataset" : "Chest X-Ray (5263 images)",
        "task"    : "binary_classification",
        "domain"  : "medical_imaging",
        "stage"   : "production",
    })

    mlflow.pytorch.log_model(model, artifact_path="resnet50_pneumonia")
    run_id = mlflow.active_run().info.run_id
    print(f"Run ID : {run_id}")

# Enregistrer dans le Registry
mlflow.register_model(
    model_uri=f"runs:/{run_id}/resnet50_pneumonia",
    name="pneumonia-detector-real"
)
print("✅ Vrai ResNet50 enregistré dans le Registry !")
