# 🫀 Heart Disease Prediction - ML Project

## 👥 Equipe do Projeto

| Nome | GitHub |
|------|--------|
| [Evaldo Galdino] | [@evaldocunhaf](https://github.com/evaldocunhaf) |
| [Lizandra Vieira] | [@lizandravieira](https://github.com/lizandravieira) |
| [Kauan Novello] | [@kauan-novello](https://github.com/kauan-novello) |
| [Sofia Saraiva] | [@Sofia-Saraiva](https://github.com/Sofia-Saraiva) |
| [Pedro Henrique Silva Souza] | [@hsspedro](https://github.com/hsspedro) |



## 🎓 Informações Acadêmicas

- **Disciplina:** Aprendizado de Máquina - 2025.2
- **Instituição:** CESAR School
- **Projeto:** Predição de Doenças Cardíacas usando Machine Learning

## 📋 Sobre o Projeto

Este projeto implementa um pipeline completo de Machine Learning para predição de doenças cardíacas, incluindo:

- **Análise Exploratória de Dados (EDA)** com visualizações interativas
- **Treinamento de múltiplos modelos** de classificação
- **Grid Search** para otimização de hiperparâmetros
- **Cross-validation** para avaliação robusta
- **Tracking de experimentos** com MLflow
- **API REST** para upload de dados
- **Dashboards** de visualização com ThingsBoard
- **Ambientes de desenvolvimento** com JupyterLab e JupyterHub

## 🏗️ Arquitetura do Projeto

```
ML-College-Project/
├── fastapi/              # API REST para upload de dados
├── mlflow/               # Tracking de experimentos ML
├── notebooks/            # Jupyter Notebooks com análises
├── jupyterhub/          # Ambiente JupyterHub compartilhado
├── jupyterlab/          # Dados do JupyterLab
├── postgres-init/       # Scripts de inicialização do banco
├── reports/             # Relatórios e documentação
└── docker-compose.yaml  # Orquestração dos serviços
```

## 🛠️ Tecnologias Utilizadas

### Machine Learning & Data Science
- Python 3.11
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn, Plotly
- MLflow

### Infraestrutura & DevOps
- Docker & Docker Compose
- FastAPI
- PostgreSQL 15
- ThingsBoard
- JupyterLab/JupyterHub
- AWS S3 (para armazenamento de artefatos)
- Snowflake (para armazenamento de dados)

## 📦 Serviços da Aplicação

| Serviço | Porta | Descrição |
|---------|-------|-----------|
| **FastAPI** | 8060 | API para upload de dados CSV |
| **MLflow UI** | 5050 | Interface de tracking de experimentos |
| **JupyterLab** | 8888 | Ambiente de desenvolvimento individual |
| **JupyterHub** | 8001 | Ambiente de desenvolvimento compartilhado |
| **ThingsBoard** | 9090 | Dashboard de visualização IoT |
| **Trendz Analytics** | 8889 | Analytics avançado do ThingsBoard |
| **PostgreSQL** | 5433 | Banco de dados |

## 🚀 Instruções de Instalação e Execução

### 📋 Pré-requisitos

- Docker Desktop instalado ([Download](https://www.docker.com/products/docker-desktop))
- Docker Compose (incluído no Docker Desktop)
- Git
- Mínimo de 8GB de RAM disponível
- 10GB de espaço em disco

### 1️⃣ Clone o Repositório

```bash
git clone https://github.com/P-E-N-T-E-S/ML-College-Project.git
cd ML-College-Project
```

### 2️⃣ Configure as Variáveis de Ambiente

Crie um arquivo `.env` na pasta raiz do projeto com as seguintes variáveis:

```bash
# Snowflake Credentials
ACCOUNT_ID=sua_conta_snowflake
USERNAME=seu_usuario
PASSWORD=sua_senha
ROLE=seu_role
WAREHOUSE=seu_warehouse
DATABASE=seu_database
SCHEMA=seu_schema

# AWS S3 Credentials
AWS_ACCESS_KEY_ID=sua_access_key
AWS_SECRET_ACCESS_KEY=sua_secret_key
AWS_DEFAULT_REGION=us-east-2
BUCKET_NAME=seu_bucket
BUCKER_DIRECTORY=data/

# MLflow
METRICS_PATH=./metrics

# ThingsBoard
JWT_TOKEN_SIGNING_KEY=sua_chave_secreta_jwt
```

**Nota:** Para desenvolvimento local, você pode omitir as credenciais do Snowflake e AWS se não for usar essas integrações.

### 3️⃣ Crie o arquivo .env do MLflow

```bash
cp .env mlflow/.env
```

### 4️⃣ Levante a Infraestrutura

```bash
# Construir e iniciar todos os serviços
docker compose up -d --build

# Ou iniciar serviços específicos
docker compose up -d fastapi mlflow jupyterlab jupyterhub
```

### 5️⃣ Verificar Status dos Containers

```bash
docker compose ps
```

Todos os serviços devem estar com status "Up".

## 📊 Acessando os Dashboards e Ferramentas

### 🔬 MLflow - Tracking de Experimentos

1. Acesse: http://localhost:5050
2. Visualize experimentos, métricas e modelos treinados
3. Compare diferentes runs e hiperparâmetros

### 📓 JupyterLab - Desenvolvimento Individual

1. Acesse: http://localhost:8888
2. Sem necessidade de token/senha
3. Notebooks disponíveis em `/work`
4. Execute as análises em `main.ipynb`

### 👥 JupyterHub - Desenvolvimento Colaborativo

1. Acesse: http://localhost:8001
2. Login: qualquer usuário (ex: `admin`)
3. Senha: deixe em branco ou digite qualquer coisa
4. Notebooks disponíveis em `/workspace`

### 📡 ThingsBoard - Dashboard IoT

1. Acesse: http://localhost:9090
2. Login padrão:
   - **Email:** tenant@thingsboard.org
   - **Senha:** tenant
3. Configure devices e dashboards para visualizar dados

### 🚀 FastAPI - API de Upload

```bash
# Testar endpoint
curl http://localhost:8060/hello

# Upload de arquivo CSV
curl -X POST "http://localhost:8060/upload" \
  -F "file=@/caminho/para/heart.csv"
```

Documentação interativa: http://localhost:8060/docs

## 📈 Executando o Pipeline de ML

### Opção 1: Via JupyterLab/JupyterHub

1. Acesse o JupyterLab (porta 8888) ou JupyterHub (porta 8001)
2. Abra o notebook `notebooks/main.ipynb`
3. Execute as células sequencialmente:
   - Importação de dados do Snowflake
   - Análise exploratória com visualizações
   - Tratamento de outliers
   - Feature engineering
   - Treinamento de modelos (7 algoritmos)
   - Grid Search para otimização
   - Cross-validation
   - Avaliação de métricas

### Opção 2: Via Script Python

```bash
# Entrar no container do MLflow
docker exec -it mlflow bash

# Executar o pipeline
python main.py
```

## 🧪 Modelos Implementados

O projeto treina e compara os seguintes modelos:

1. **Logistic Regression**
2. **Decision Tree**
3. **Random Forest**
4. **Gradient Boosting**
5. **Support Vector Machine (SVM)**
6. **Gaussian Naive Bayes**
7. **K-Nearest Neighbors (KNN)**

Cada modelo passa por:
- Grid Search para otimização de hiperparâmetros
- Cross-validation (5 folds)
- Avaliação com múltiplas métricas:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Specificity
  - AUC-ROC

## 📊 Visualizações Disponíveis

O notebook inclui visualizações completas:

- 📊 Distribuição das variáveis target
- 📈 Histogramas de variáveis numéricas
- 📦 Boxplots por classe
- 🔥 Matriz de correlação
- 🎯 Top correlações com target
- 📉 Scatter plots interativos
- 🎻 Violin plots por grupo
- 🔄 Pairplots multivariados
- 🧮 Matriz de confusão
- ⭐ Feature importance

## 🛑 Parando os Serviços

```bash
# Parar todos os serviços
docker compose down

# Parar e remover volumes (ATENÇÃO: apaga dados persistidos)
docker compose down -v

# Parar apenas serviços específicos
docker compose stop mlflow jupyterlab
```

## 🔄 Reiniciando os Containers

```bash
# Script de reinicialização
./restart-containers.sh

# Ou manualmente
docker compose restart
```

## 🐛 Troubleshooting

### Erro: "Port already in use"

```bash
# Verificar portas em uso
lsof -i :8888  # ou a porta específica

# Parar containers conflitantes
docker compose down
```

### Erro: "Platform mismatch (linux/amd64 vs linux/arm64)"

Já resolvido no docker-compose.yaml com `platform: linux/amd64`

### Erro: "Cannot connect to Snowflake"

Verifique as credenciais no arquivo `.env` e `mlflow/.env`

### Containers não iniciam

```bash
# Ver logs detalhados
docker compose logs -f [nome_do_servico]

# Reconstruir imagens
docker compose up -d --build --force-recreate
```

## 📝 Estrutura de Dados

O projeto utiliza o dataset **Heart Disease** com as seguintes features:

- **age:** Idade do paciente
- **sex:** Sexo (0=F, 1=M)
- **cp:** Tipo de dor no peito (0-3)
- **trestbps:** Pressão arterial em repouso
- **chol:** Colesterol sérico
- **fbs:** Glicemia em jejum > 120 mg/dl
- **restecg:** Resultados eletrocardiográficos
- **thalach:** Frequência cardíaca máxima alcançada
- **exang:** Angina induzida por exercício
- **oldpeak:** Depressão ST induzida por exercício
- **slope:** Inclinação do segmento ST no exercício
- **ca:** Número de vasos principais (0-3)
- **thal:** Talassemia (0=normal; 1=defeito fixo; 2=defeito reversível)
- **target:** Presença de doença cardíaca (0=Não, 1=Sim)

## 🤝 Contribuindo

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença especificada no arquivo [LICENSE](LICENSE).

## 📞 Contato

Para dúvidas ou sugestões, entre em contato com a equipe através do GitHub.

---

**Desenvolvido com ❤️ para CESAR School | Aprendizado de Máquina 2025.2**
