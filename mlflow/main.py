import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import snowflake.connector
from utils import *

from dotenv import load_dotenv
import os
from mlflow.models.signature import infer_signature
from datetime import datetime

load_dotenv()

ACCOUNT_ID = os.getenv("ACCOUNT_ID")
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")
ROLE = os.getenv("ROLE")
WAREHOUSE = os.getenv("WAREHOUSE")
DATABASE = os.getenv("DATABASE")
SCHEMA = os.getenv("SCHEMA")
METRICS_PATH = os.getenv("METRICS_PATH")

# Configurar credenciais AWS para acesso ao S3
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
os.environ["AWS_DEFAULT_REGION"] = os.getenv("AWS_DEFAULT_REGION", "us-east-2")

# Verificar se as credenciais AWS foram carregadas
print(f"AWS_ACCESS_KEY_ID configurado: {bool(os.environ.get('AWS_ACCESS_KEY_ID'))}")
print(f"AWS Region: {os.environ.get('AWS_DEFAULT_REGION')}")

# ============================
#  CONFIGURAÇÃO PRINCIPAL
# ============================

# Usar SQLite como backend de tracking e S3 para artefatos
# Configurar tracking URI com SQLite
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Configurar o experimento
experiment_name = "heart-disease-ml"
experiment = mlflow.get_experiment_by_name(experiment_name)

# Se o experimento já existe mas não tem artifact_location correto, deletar e recriar
if experiment is not None:
    if not experiment.artifact_location.startswith("s3://"):
        print(f"Experimento existente tem artifact_location incorreto: {experiment.artifact_location}")
        print("Deletando experimento antigo...")
        mlflow.delete_experiment(experiment.experiment_id)
        experiment = None

if experiment is None:
    # Criar experimento com artifact location no S3
    experiment_id = mlflow.create_experiment(
        experiment_name,
        artifact_location="s3://project-ml-college/mlflow-artifacts/"
    )
    print(f"Experimento criado com ID: {experiment_id} e artifact_location: s3://project-ml-college/mlflow-artifacts/")
else:
    experiment_id = experiment.experiment_id
    print(f"Usando experimento existente com ID: {experiment_id}")
    print(f"Artifact location: {experiment.artifact_location}")

# Ativar o experimento
mlflow.set_experiment(experiment_name)

conn = snowflake.connector.connect(
    account= ACCOUNT_ID,  
    user= USERNAME,        
    password= PASSWORD,
    role= ROLE,             
    warehouse= WAREHOUSE,     
    database= DATABASE,
    schema= SCHEMA    
)

print("Conexão com o Snowflake estabelecida com sucesso!")

df = pd.read_sql("SELECT * FROM dados_modelo", conn)

print("Dados carregados com sucesso!")

df.columns = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","target"]

outlier_columns = ["trestbps", "chol"]

num_columns = ["age", "sex", "trestbps", "chol", "thalach", "oldpeak"]

for column in outlier_columns:
    lower_limit, upper_limit = get_fences(df, column)
    df = df.loc[(df[column] >= lower_limit) & (df[column] <= upper_limit)]

scaler = StandardScaler()
df[num_columns] = scaler.fit_transform(df[num_columns])
df.head()

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

models = {
"Logistic Regression": LogisticRegression(C=0.01, max_iter=100, solver='liblinear'),
"Decision Tree": DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_split=2, min_samples_leaf=1),
"Random Forest": RandomForestClassifier(max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=200),
"Gradient Boosting": GradientBoostingClassifier(learning_rate=0.2, n_estimators=200, max_depth=5, min_samples_split=10),
"Support Vector Machine": SVC(probability=True, C=100, gamma='scale', kernel='rbf'),
"Gaussian Naive Bayes": GaussianNB(var_smoothing=1e-09),
"K-Nearest Neighbors": KNeighborsClassifier(metric="manhattan", n_neighbors=7, weights="distance"),
}

print("Modelos definidos com sucesso!")

cv_results = []

# Adicionar timestamp para forçar novos runs
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for model_name, best_model in models.items():

    run_name_with_time = f"{model_name}_{timestamp}"
    
    with mlflow.start_run(run_name=run_name_with_time):
    
        # Treinar o modelo
        print("Treinando o modelo: {}".format(model_name))
        best_model.fit(X_train, y_train)
        
        # Fazer predições com cross validation
        print("Fazendo predições com cross validation para o modelo: {}".format(model_name))
        y_pred_cv = cross_val_predict(best_model, X_train, y_train, cv=5)
        
        # Calcular métricas
        metrics = calculate_metrics(y_train, y_pred_cv)
        
        cv_results.append(metrics)

        # Printar métricas
        print(f"\nMétricas para {model_name}:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
        print()

        # Logar métricas individualmente
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Criar exemplo de input e assinatura do modelo para MLflow
        X_example = X_train.head(1)
        try:
            signature = infer_signature(X_example, best_model.predict(X_example))
        except Exception:
            signature = None

        # Salvar modelo com signature e input_example usando o nome do modelo
        model_artifact_name = model_name.lower().replace(" ", "_")
        
        # Obter informações do run atual
        run_id = mlflow.active_run().info.run_id
        run_info = mlflow.active_run().info
        
        print(f"\n{'='*60}")
        print(f"Salvando modelo: {model_name}")
        print(f"Run ID: {run_id}")
        print(f"Artifact URI do experimento: {run_info.artifact_uri}")
        
        # Log do modelo para o S3
        try:
            model_info = mlflow.sklearn.log_model(
                sk_model=best_model, 
                artifact_path=model_artifact_name,
                signature=signature, 
                input_example=X_example
            )
            print(f"✓ Modelo salvo com sucesso!")
            print(f"  Model URI: {model_info.model_uri}")
            print(f"  Artifact path: {model_artifact_name}")
        except Exception as e:
            print(f"✗ Erro ao salvar modelo: {str(e)}")
            raise
        
        print(f"{'='*60}\n")

print("Métricas salvas localmente e modelo enviado para S3!")
