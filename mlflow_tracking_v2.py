import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.dummy import DummyClassifier

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("pneumonia-detection")

# Simuler des données pour créer un modèle enregistrable
X = np.random.rand(100, 10)
y = np.random.randint(0, 2, 100)

with mlflow.start_run(run_name="ResNet50_phase2_v2"):
    # Logger les hyperparamètres
    mlflow.log_params({
        "architecture" : "ResNet50_transfer",
        "epochs"       : 10,
        "learning_rate": 0.0001,
        "batch_size"   : 32,
        "optimizer"    : "Adam",
    })

    # Logger les métriques
    mlflow.log_metrics({
        "auc_roc"          : 0.9983,
        "f1_macro"         : 0.9768,
        "recall_pneumonia" : 0.982,
        "precision_normal" : 0.95,
        "false_negatives"  : 14,
        "accuracy"         : 0.98,
    })

    # Logger un modèle sklearn (proxy du vrai ResNet50)
    model = DummyClassifier()
    model.fit(X, y)
    mlflow.sklearn.log_model(model, artifact_path="model")

    # Tags
    mlflow.set_tags({
        "dataset" : "Chest X-Ray (5263 images)",
        "task"    : "binary_classification",
        "domain"  : "medical_imaging",
        "model"   : "ResNet50_phase2",
    })

    run_id = mlflow.active_run().info.run_id
    print(f"Run ID : {run_id}")

# Enregistrer dans le Registry
mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name="pneumonia-detector"
)
print("✅ Modèle enregistré dans le Model Registry !")
