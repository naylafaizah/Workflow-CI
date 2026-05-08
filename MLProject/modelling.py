import mlflow
import mlflow.sklearn
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score,
                             roc_auc_score, confusion_matrix)

# Setup tracking ke DagsHub via environment variable
# (di-set oleh GitHub Actions dari secrets)
mlflow.set_tracking_uri(
    f"https://dagshub.com/naylafffz/mlsystem-diabetes.mlflow"
)

# Parameter
n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
max_depth    = int(sys.argv[2]) if len(sys.argv) > 2 else 10
if max_depth == 0:
    max_depth = None

# Load data
X_train = pd.read_csv('diabetes_preprocessing/X_train.csv')
X_test  = pd.read_csv('diabetes_preprocessing/X_test.csv')
y_train = pd.read_csv('diabetes_preprocessing/y_train.csv').squeeze()
y_test  = pd.read_csv('diabetes_preprocessing/y_test.csv').squeeze()

mlflow.set_experiment('Diabetes-CI')

with mlflow.start_run(run_name=f'CI-RF-{n_estimators}-{max_depth}'):
    # Training
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Log
    mlflow.log_param('n_estimators', n_estimators)
    mlflow.log_param('max_depth',    max_depth)
    mlflow.log_metric('accuracy',    accuracy_score(y_test, y_pred))
    mlflow.log_metric('f1_score',    f1_score(y_test, y_pred))
    mlflow.log_metric('precision',   precision_score(y_test, y_pred))
    mlflow.log_metric('recall',      recall_score(y_test, y_pred))
    mlflow.log_metric('roc_auc',     roc_auc_score(y_test, y_prob))

    # Artifact confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    mlflow.log_artifact('confusion_matrix.png')
    mlflow.sklearn.log_model(model, 'model')

    run_id = mlflow.active_run().info.run_id
    with open('latest_run_id.txt', 'w') as f:
        f.write(run_id)

    print(f'Run ID   : {run_id}')
    print(f'Accuracy : {accuracy_score(y_test, y_pred):.4f}')
    print(f'F1 Score : {f1_score(y_test, y_pred):.4f}')