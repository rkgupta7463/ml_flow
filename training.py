import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load data
X, y = load_iris(return_X_y=True)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Start MLflow run
with mlflow.start_run() as run:
    # Log model and register it
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="iris_model",
        registered_model_name="IrisClassifier"
    )
    run_id = run.info.run_id
    print(f"Model logged under run_id={run_id}")

# Transition latest model to Production automatically
client = MlflowClient()
latest_version = client.get_latest_versions("IrisClassifier", stages=["None"])[0].version

client.transition_model_version_stage(
    name="IrisClassifier",
    version=latest_version,
    stage="Production",
    archive_existing_versions=True  # optional: moves old production models to Archived
)

print(f"Model IrisClassifier version {latest_version} moved to Production ✅")
