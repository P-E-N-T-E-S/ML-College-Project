"""
Simulador de streaming de dados
Lê dados do Snowflake e envia para a API em tempo real
"""

import sys
import os

# Adicionar paths corretos
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/src')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import snowflake.connector
import pickle
import boto3
from botocore.client import Config
from io import BytesIO
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

# Carregar variáveis de ambiente
load_dotenv()

# Credenciais Snowflake
ACCOUNT_ID = os.getenv("ACCOUNT_ID")
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")
ROLE = os.getenv("ROLE")
WAREHOUSE = os.getenv("WAREHOUSE")
DATABASE = os.getenv("DATABASE")
SCHEMA = os.getenv("SCHEMA")

# Credenciais AWS S3
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION") or os.getenv("REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET") or os.getenv("BUCKET_NAME")
S3_PREFIX = os.getenv("S3_PREFIX") or os.getenv("BUCKET_DIRECTORY", "models/")

# Debug: Printar variáveis de ambiente carregadas
print(f"🔍 DEBUG - Variáveis de ambiente:")
print(f"  ACCOUNT_ID: {'✅' if ACCOUNT_ID else '❌'} {ACCOUNT_ID}")
print(f"  USERNAME: {'✅' if USERNAME else '❌'} {USERNAME}")
print(f"  PASSWORD: {'✅' if PASSWORD else '❌'} {'***' if PASSWORD else None}")
print(f"  ROLE: {'✅' if ROLE else '❌'} {ROLE}")
print(f"  WAREHOUSE: {'✅' if WAREHOUSE else '❌'} {WAREHOUSE}")
print(f"  DATABASE: {'✅' if DATABASE else '❌'} {DATABASE}")
print(f"  SCHEMA: {'✅' if SCHEMA else '❌'} {SCHEMA}")
print(f"  AWS_ACCESS_KEY_ID: {'✅' if AWS_ACCESS_KEY_ID else '❌'} {AWS_ACCESS_KEY_ID[:10] + '...' if AWS_ACCESS_KEY_ID else None}")
print(f"  AWS_SECRET_ACCESS_KEY: {'✅' if AWS_SECRET_ACCESS_KEY else '❌'} {'***' if AWS_SECRET_ACCESS_KEY else None}")
print(f"  AWS_REGION: {'✅' if AWS_REGION else '❌'} {AWS_REGION}")
print(f"  S3_BUCKET: {'✅' if S3_BUCKET else '❌'} {S3_BUCKET}")
print(f"  S3_PREFIX: {'✅' if S3_PREFIX else '❌'} {S3_PREFIX}")

# Importar ThingsBoard Client
try:
    from thingsboard_client import ThingsBoardClient
except ImportError:
    ThingsBoardClient = None
    print("⚠️ ThingsBoard client não disponível. Continuando sem integração...")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StreamingSimulator:
    def __init__(self, api_url: str = "http://api:8000", delay: float = 2.0, 
                 thingsboard_token: str = None):
        self.api_url = api_url
        self.delay = delay
        
        # Inicializar conexão Snowflake
        try:
            # Validar variáveis obrigatórias
            required_vars = {
                'ACCOUNT_ID': ACCOUNT_ID,
                'USERNAME': USERNAME,
                'PASSWORD': PASSWORD,
                'ROLE': ROLE,
                'WAREHOUSE': WAREHOUSE,
                'DATABASE': DATABASE,
                'SCHEMA': SCHEMA
            }
            
            missing_vars = [k for k, v in required_vars.items() if not v]
            if missing_vars:
                logger.error(f"❌ Variáveis de ambiente faltando: {', '.join(missing_vars)}")
                self.snowflake_conn = None
            else:
                self.snowflake_conn = snowflake.connector.connect(
                    account=ACCOUNT_ID,
                    user=USERNAME,
                    password=PASSWORD,
                    role=ROLE,
                    warehouse=WAREHOUSE,
                    database=DATABASE,
                    schema=SCHEMA
                )
                logger.info("✅ Conexão com Snowflake estabelecida")
        except Exception as e:
            logger.error(f"❌ Erro ao conectar ao Snowflake: {str(e)}")
            self.snowflake_conn = None
        
        # Inicializar cliente S3
        try:
            # Validar variáveis S3 obrigatórias
            s3_required_vars = {
                'AWS_ACCESS_KEY_ID': AWS_ACCESS_KEY_ID,
                'AWS_SECRET_ACCESS_KEY': AWS_SECRET_ACCESS_KEY,
                'S3_BUCKET': S3_BUCKET
            }
            
            missing_s3_vars = [k for k, v in s3_required_vars.items() if not v]
            if missing_s3_vars:
                logger.warning(f"⚠️ Variáveis S3 faltando: {', '.join(missing_s3_vars)}. Cliente S3 não será inicializado.")
                self.s3_client = None
            else:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                    region_name=AWS_REGION,
                    config=Config(signature_version='s3v4')
                )
                logger.info("✅ Cliente S3 inicializado")
        except Exception as e:
            logger.error(f"❌ Erro ao inicializar S3: {str(e)}")
            self.s3_client = None
        
        # Inicializar ThingsBoard se token fornecido
        self.tb_client = None
        if thingsboard_token and ThingsBoardClient:
            self.tb_client = ThingsBoardClient(
                host="http://thingsboard:9090",
                access_token=thingsboard_token
            )
            logger.info("✅ ThingsBoard client inicializado")
        elif thingsboard_token:
            logger.warning("⚠️ Token fornecido mas ThingsBoard client não disponível")
        
    def load_data_from_snowflake(self):
        """Carrega dados do Snowflake"""
        logger.info("📥 Carregando dados do Snowflake...")
        
        if not self.snowflake_conn:
            logger.error("❌ Conexão com Snowflake não disponível")
            return None
        
        try:
            # Carregar dados da tabela do Snowflake
            query = "SELECT * FROM dados_modelo"
            df = pd.read_sql(query, self.snowflake_conn)
            
            logger.info(f"✅ {len(df)} registros carregados do Snowflake")
            return df
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar dados do Snowflake: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_validation_data(self):
        """Carrega e prepara dados de validação do Snowflake"""
        logger.info("📥 Preparando dados de validação...")
        
        try:
            # Carregar dados do Snowflake
            df_raw = self.load_data_from_snowflake()
            
            if df_raw is None:
                return None, None
            
            # Renomear colunas para o formato esperado
            df = df_raw.copy()
            df.columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", 
                         "restecg", "thalach", "exang", "oldpeak", "slope", 
                         "ca", "thal", "target"]
            
            # Separar features e target
            X = df.drop('target', axis=1)
            y = df['target']
            
            # Usar uma parte dos dados para validação (30%)
            
            _, X_val, _, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
            
            logger.info(f"✅ {len(X_val)} amostras preparadas para validação")
            return X_val, y_val
            
        except Exception as e:
            logger.error(f"❌ Erro ao preparar dados: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def load_model_from_s3(self, s3_path: str):
        """Carrega modelo do S3"""
        try:
            logger.info(f"📥 Carregando modelo do S3: s3://{S3_BUCKET}/{s3_path}")
            
            # Baixar arquivo do S3
            response = self.s3_client.get_object(Bucket=S3_BUCKET, Key=s3_path)
            model_data = response['Body'].read()
            
            # Deserializar modelo
            model = pickle.loads(model_data)
            
            logger.info("✅ Modelo carregado com sucesso do S3")
            return model
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo do S3: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_production_model(self):
        """Carrega o modelo de produção do S3"""
        try:
            # Validar se S3 está configurado
            if not self.s3_client:
                logger.error("❌ Cliente S3 não inicializado. Verifique as variáveis de ambiente.")
                return None, None
            
            if not S3_BUCKET:
                logger.error("❌ S3_BUCKET não configurado.")
                return None, None
            
            # Listar objetos no bucket para encontrar o modelo de produção
            response = self.s3_client.list_objects_v2(
                Bucket=S3_BUCKET,
                Prefix=S3_PREFIX
            )
            
            if 'Contents' not in response:
                logger.error("❌ Nenhum modelo encontrado no S3")
                return None, None
            
            # Encontrar modelo mais recente
            model_files = [obj for obj in response['Contents'] 
                          if obj['Key'].endswith('.pkl')]
            
            if not model_files:
                logger.error("❌ Nenhum arquivo .pkl encontrado no S3")
                return None, None
            
            # Ordenar por data de modificação (mais recente primeiro)
            latest_model = max(model_files, key=lambda x: x['LastModified'])
            model_path = latest_model['Key']
            
            logger.info(f"📦 Modelo de produção: {model_path}")
            
            # Carregar modelo
            model = self.load_model_from_s3(model_path)
            
            # Metadata
            metadata = {
                'model_name': model_path.split('/')[-1].replace('.pkl', ''),
                'last_modified': latest_model['LastModified'].isoformat(),
                'size': latest_model['Size']
            }
            
            return model, metadata
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo de produção: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def __del__(self):
        """Fecha conexão com Snowflake ao destruir objeto"""
        if hasattr(self, 'snowflake_conn') and self.snowflake_conn:
            try:
                self.snowflake_conn.close()
                logger.info("✅ Conexão com Snowflake fechada")
            except:
                pass
    
    def convert_to_api_format(self, row: pd.Series) -> dict:
        """Converte linha do DataFrame para formato esperado pela API"""
        # A API espera os nomes no formato snake_case conforme PatientData
        return {
            'age': float(row.get('age', 0)),
            'sex': int(row.get('sex', 0)),
            'chest_pain_type': int(row.get('chest pain type', 0)),
            'resting_bp': float(row.get('resting bp s', 0)),
            'cholesterol': float(row.get('cholesterol', 0)),
            'fasting_bs': int(row.get('fasting blood sugar', 0)),
            'resting_ecg': int(row.get('resting ecg', 0)),
            'max_hr': float(row.get('max heart rate', 0)),
            'exercise_angina': int(row.get('exercise angina', 0)),
            'oldpeak': float(row.get('oldpeak', 0)),
            'st_slope': int(row.get('ST slope', 0))
        }
    
    def send_prediction_request(self, patient_data: dict, patient_id: int, true_label: int):
        """Envia requisição de predição para a API"""
        try:
            # Enviar requisição para endpoint /predict
            response = requests.post(
                f"{self.api_url}/predict",
                json=patient_data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                return result
            else:
                logger.error(f"❌ Erro na API: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erro de conexão: {str(e)}")
            return None
    
    def run(self, max_samples: int = None, use_api: bool = True):
        """
        Executa simulação de streaming
        
        Args:
            max_samples: Número máximo de amostras (None = todas)
            use_api: Se True, usa API. Se False, usa modelo local direto do S3
        """
        logger.info("🚀 Iniciando simulação de streaming...")
        logger.info(f"⏱️  Delay entre requisições: {self.delay}s")
        logger.info(f"🎯 Modo: {'API' if use_api else 'Modelo Direto (S3)'}")
        
        # Carregar dados
        X_val, y_val = self.load_validation_data()
        
        if X_val is None:
            logger.error("❌ Não foi possível carregar dados. Abortando.")
            return
        
        # Se não usar API, carregar modelo direto
        model = None
        model_metadata = None
        if not use_api:
            model, model_metadata = self.load_production_model()
            if model is None:
                logger.error("❌ Não foi possível carregar modelo. Abortando.")
                return
        
        # Limitar amostras se especificado
        if max_samples:
            X_val = X_val.head(max_samples)
            y_val = y_val.head(max_samples)
        
        total = len(X_val)
        correct_predictions = 0
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 INICIANDO STREAMING DE {total} PACIENTES")
        logger.info(f"{'='*60}\n")
        
        # Processar cada amostra
        for idx, (_, row) in enumerate(X_val.iterrows(), 1):
            patient_id = idx
            true_label = int(y_val.iloc[idx - 1])
            
            logger.info(f"\n{'─'*60}")
            logger.info(f"🏥 Paciente {patient_id}/{total}")
            logger.info(f"📋 Label Real: {'Doença ❤️‍🩹' if true_label == 1 else 'Saudável ✅'}")
            
            if use_api:
                # Usar API
                patient_data = self.convert_to_api_format(row)
                result = self.send_prediction_request(patient_data, patient_id, true_label)
                
                if result:
                    predicted_label = result.get('prediction')
                    probability = result.get('probability', 0.0)
                    model_name = result.get('model_name', 'unknown')
                else:
                    logger.warning(f"⚠️ Falha na predição do paciente {patient_id}")
                    continue
            else:
                # Usar modelo direto do S3
                try:
                    # Preparar dados
                    X = pd.DataFrame([row])
                    
                    # Predição
                    predicted_label = int(model.predict(X)[0])
                    
                    # Probabilidade
                    if hasattr(model, 'predict_proba'):
                        probability = float(model.predict_proba(X)[0][1])
                    else:
                        probability = float(predicted_label)
                    
                    model_name = model_metadata['model_name']
                    
                except Exception as e:
                    logger.error(f"❌ Erro na predição: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Avaliar resultado
            is_correct = (predicted_label == true_label)
            if is_correct:
                correct_predictions += 1
            
            # Log resultado
            emoji = "✅" if is_correct else "❌"
            logger.info(f"🔮 Predição: {'Doença ❤️‍🩹' if predicted_label == 1 else 'Saudável ✅'}")
            logger.info(f"📊 Probabilidade: {probability:.2%}")
            logger.info(f"🤖 Modelo: {model_name}")
            logger.info(f"{emoji} {'CORRETO' if is_correct else 'INCORRETO'}")
            logger.info(f"📈 Acurácia Atual: {correct_predictions}/{idx} ({correct_predictions/idx*100:.1f}%)")
            
            # Enviar para ThingsBoard se configurado
            if self.tb_client:
                self.tb_client.send_prediction(
                    patient_id=patient_id,
                    prediction=predicted_label,
                    probability=probability,
                    true_label=true_label,
                    is_correct=is_correct,
                    model_name=model_name
                )
            
            # Delay antes da próxima amostra
            if idx < total:
                time.sleep(self.delay)
        
        # Resumo final
        final_accuracy = correct_predictions / total * 100 if total > 0 else 0
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 RESUMO FINAL")
        logger.info(f"{'='*60}")
        logger.info(f"✅ Total de amostras: {total}")
        logger.info(f"✅ Predições corretas: {correct_predictions}")
        logger.info(f"❌ Predições incorretas: {total - correct_predictions}")
        logger.info(f"📈 Acurácia Final: {final_accuracy:.2f}%")
        logger.info(f"{'='*60}\n")
        
        # Enviar resumo final para ThingsBoard
        if self.tb_client:
            self.tb_client.send_summary(
                total=total,
                correct=correct_predictions,
                accuracy=final_accuracy
            )
            logger.info("📊 Resumo enviado para ThingsBoard")

def wait_for_api(api_url: str, max_retries: int = 30):
    """Aguarda API estar pronta"""
    logger.info("⏳ Aguardando API estar pronta...")
    
    for i in range(max_retries):
        try:
            response = requests.get(f"{api_url}/health", timeout=2)
            if response.status_code == 200:
                logger.info("✅ API está pronta!")
                return True
        except:
            pass
        
        if i < max_retries - 1:
            time.sleep(2)
    
    logger.error(f"❌ API não está respondendo após {max_retries * 2}s")
    return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Simulador de Streaming de Dados')
    parser.add_argument('--api-url', default='http://api:8000', help='URL da API')
    parser.add_argument('--delay', type=float, default=2.0, help='Delay entre requisições (segundos)')
    parser.add_argument('--max-samples', type=int, default=None, help='Número máximo de amostras')
    parser.add_argument('--no-api', action='store_true', help='Não usar API, carregar modelo direto do S3')
    parser.add_argument('--tb-token', default=None, help='ThingsBoard device access token')
    
    args = parser.parse_args()
    
    # Pegar token do ambiente se não fornecido
    tb_token = args.tb_token or os.getenv('THINGSBOARD_TOKEN')
    
    # Debug token
    if tb_token:
        logger.info(f"🔑 ThingsBoard Token: {tb_token[:10]}...{tb_token[-10:] if len(tb_token) > 20 else ''}")
    else:
        logger.warning("⚠️ Nenhum token do ThingsBoard configurado")
    
    use_api = not args.no_api
    
    # Se usar API, aguardar estar pronta
    if use_api:
        if not wait_for_api(args.api_url):
            logger.warning("⚠️ API não está respondendo. Tentando com modelo direto do S3...")
            use_api = False
    
    # Iniciar simulação
    simulator = StreamingSimulator(
        api_url=args.api_url, 
        delay=args.delay,
        thingsboard_token=tb_token
    )
    simulator.run(max_samples=args.max_samples, use_api=use_api)