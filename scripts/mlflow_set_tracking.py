import mlflow

mlflow.set_tracking_uri(
    "azureml://eastus.api.azureml.ms/mlflow/v1.0/subscriptions/8702bce1-ac17-444a-b392-3cc0530b1276/resourceGroups/carbonCruncherRG/providers/Microsoft.MachineLearningServices/workspaces/carbonCruncherWS"
)

print("Tracking server set!")
