#!/bin/bash

# Carregar variáveis do .env
export $(grep -v '^#' .env | xargs)

# Iniciar MLflow UI com as credenciais AWS
mlflow ui --backend-store-uri ./metrics --port 5000
