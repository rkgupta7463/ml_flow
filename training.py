# import mlflow
# import mlflow.sklearn
# from mlflow.tracking import MlflowClient
# from sklearn.datasets import load_iris
# from sklearn.linear_model import LogisticRegression

# # Load data
# X, y = load_iris(return_X_y=True)

# # Train model
# model = LogisticRegression(max_iter=200)
# model.fit(X, y)

# # Start MLflow run
# with mlflow.start_run() as run:
#     # Log model and register it
#     mlflow.sklearn.log_model(
#         sk_model=model,
#         artifact_path="iris_model",
#         registered_model_name="IrisClassifier"
#     )
#     run_id = run.info.run_id
#     print(f"Model logged under run_id={run_id}")

# # Transition latest model to Production automatically
# client = MlflowClient()
# latest_version = client.get_latest_versions("IrisClassifier", stages=["None"])[0].version

# client.transition_model_version_stage(
#     name="IrisClassifier",
#     version=latest_version,
#     stage="Production",
#     archive_existing_versions=True  # optional: moves old production models to Archived
# )

# print(f"Model IrisClassifier version {latest_version} moved to Production ✅")


import mlflow
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

# set the experiment id
mlflow.set_experiment(experiment_id="692432011859519388")

mlflow.autolog()
db = load_diabetes()

X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

# Create and train models.
rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
rf.fit(X_train, y_train)

# Use the model to make predictions on the test dataset.
predictions = rf.predict(X_test)
