"""
Cliente ThingsBoard para envio de telemetria
"""

import requests
import json
import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class ThingsBoardClient:
    def __init__(self, host: str = "http://thingsboard:9090", access_token: str = None):
        """
        Inicializa cliente ThingsBoard
        
        Args:
            host: URL do ThingsBoard (padrão: http://thingsboard:9090)
            access_token: Token de acesso do device
        """
        self.host = host
        self.access_token = access_token
        self.telemetry_url = f"{host}/api/v1/{access_token}/telemetry"
        self.attributes_url = f"{host}/api/v1/{access_token}/attributes"
        
        if not access_token:
            logger.warning("⚠️ ThingsBoard access token não configurado")
    
    def send_telemetry(self, data: Dict[str, Any]) -> bool:
        """
        Envia dados de telemetria para o ThingsBoard
        
        Args:
            data: Dicionário com dados a enviar
            
        Returns:
            True se sucesso, False caso contrário
        """
        if not self.access_token:
            return False
        
        try:
            response = requests.post(
                self.telemetry_url,
                json=data,
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            
            if response.status_code == 200:
                logger.debug(f"✅ Telemetria enviada: {list(data.keys())}")
                return True
            else:
                logger.error(f"❌ Erro ao enviar telemetria: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erro ao enviar telemetria: {str(e)}")
            return False
    
    def send_attributes(self, data: Dict[str, Any]) -> bool:
        """
        Envia atributos do device
        
        Args:
            data: Dicionário com atributos
            
        Returns:
            True se sucesso, False caso contrário
        """
        if not self.access_token:
            return False
        
        try:
            response = requests.post(
                self.attributes_url,
                json=data,
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            
            if response.status_code == 200:
                logger.debug(f"✅ Atributos enviados: {list(data.keys())}")
                return True
            else:
                logger.error(f"❌ Erro ao enviar atributos: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erro ao enviar atributos: {str(e)}")
            return False
    
    def send_prediction(self, patient_id: int, prediction: int, probability: float,
                       true_label: int, is_correct: bool, model_name: str) -> bool:
        """
        Envia dados de uma predição
        
        Args:
            patient_id: ID do paciente
            prediction: Predição (0 ou 1)
            probability: Probabilidade da classe 1
            true_label: Label verdadeiro
            is_correct: Se a predição estava correta
            model_name: Nome do modelo
            
        Returns:
            True se sucesso
        """
        telemetry = {
            "patient_id": patient_id,
            "prediction": prediction,
            "probability": probability * 100,  # Converter para porcentagem
            "true_label": true_label,
            "is_correct": 1 if is_correct else 0,
            "timestamp": int(datetime.now().timestamp() * 1000)  # Timestamp em ms
        }
        
        # Enviar atributos do modelo (apenas uma vez ou quando mudar)
        attributes = {
            "model_name": model_name
        }
        
        self.send_attributes(attributes)
        return self.send_telemetry(telemetry)
    
    def send_summary(self, total: int, correct: int, accuracy: float) -> bool:
        """
        Envia resumo das predições
        
        Args:
            total: Total de predições
            correct: Predições corretas
            accuracy: Acurácia (0-100)
            
        Returns:
            True se sucesso
        """
        telemetry = {
            "total_predictions": total,
            "correct_predictions": correct,
            "accuracy": accuracy,
            "timestamp": int(datetime.now().timestamp() * 1000)
        }
        
        return self.send_telemetry(telemetry)