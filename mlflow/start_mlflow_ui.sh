#!/bin/bash

# Carregar variáveis do .env
export $(grep -v '^#' .env | xargs)

# Iniciar MLflow UI com SQLite e artefatos no S3
mlflow ui \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root s3://project-ml-college/mlflow-artifacts/ \
  --host 0.0.0.0 \
  --port 5000
