import requests
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import logging
import time
from typing import List, Dict, Optional, Any
from tqdm import tqdm

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaAPI:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        logger.info(f"Inicializando API do Ollama em {base_url}")
        
    def list_models(self) -> List[str]:
        """Get list of available models from Ollama."""
        try:
            logger.info("Buscando modelos disponíveis no Ollama")
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                logger.info(f"Modelos encontrados: {', '.join(model_names)}")
                return model_names
            logger.warning("Não foi possível obter lista de modelos, usando modelo padrão 'mistral'")
            return ['mistral']
        except Exception as e:
            logger.error(f"Erro ao buscar modelos: {str(e)}")
            return ['mistral']

    def process_with_model(self, text: str, model_name: str, system_prompt: Optional[str] = None) -> str:
        """Processa texto com o modelo Ollama selecionado."""
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": model_name,
            "prompt": text,
            "system": system_prompt if system_prompt else "",
            "stream": False
        }
        
        try:
            logger.info(f"\nEnviando requisição para o Ollama (modelo: {model_name})")
            logger.info(f"Tamanho do prompt: {len(text)} caracteres")
            logger.info(f"Primeiros 200 caracteres do prompt: {text[:200]}...")
            
            start_time = time.time()
            response = requests.post(url, json=payload)
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()['response']
                logger.info(f"Resposta recebida em {processing_time:.2f} segundos")
                logger.info(f"Tamanho da resposta: {len(result)} caracteres")
                return result
            else:
                error_msg = f"Erro na API do Ollama: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return error_msg
        except Exception as e:
            error_msg = f"Erro ao conectar com Ollama: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def ask_question_with_context(self, question: str, context: str, model_name: str) -> str:
        """Tenta responder a pergunta usando apenas o contexto dos documentos."""
        logger.info("Processando pergunta com contexto")
        system_prompt = """Você é um assistente especializado. Primeiro, tente responder usando apenas 
        as informações do contexto fornecido. Se a informação necessária não estiver no contexto, 
        indique explicitamente que vai usar seu conhecimento geral para responder."""
        
        prompt = f"""Contexto dos documentos:
        {context}

        Pergunta: {question}

        Por favor:
        1. Primeiro, verifique se a resposta está no contexto fornecido
        2. Se encontrar a resposta no contexto, responda usando apenas essas informações
        3. Se a resposta não estiver no contexto, indique explicitamente que vai usar seu conhecimento geral
        
        Resposta:"""
        
        return self.process_with_model(
            text=prompt,
            model_name=model_name,
            system_prompt=system_prompt
        )

    def ask_question_general(self, question: str, model_name: str) -> str:
        """Faz uma pergunta usando apenas o conhecimento base do modelo."""
        logger.info("Processando pergunta com conhecimento base")
        system_prompt = """Você é um assistente geral. Use seu conhecimento base para responder 
        à pergunta da melhor forma possível."""
        
        prompt = f"""Pergunta: {question}

        Por favor, use seu conhecimento geral para fornecer a melhor resposta possível.
        
        Resposta:"""
        
        return self.process_with_model(
            text=prompt,
            model_name=model_name,
            system_prompt=system_prompt
        )

    def ask_question(self, question: str, model_name: str, context: Optional[str] = None) -> str:
        """
        Processo de duas etapas para responder perguntas:
        1. Tenta responder usando o contexto dos documentos
        2. Se necessário, usa o conhecimento base do modelo
        """
        logger.info(f"\nProcessando pergunta: {question}")
        logger.info(f"Modelo selecionado: {model_name}")
        start_time = time.time()
        
        if context:
            logger.info("\nTentando responder com o contexto dos documentos...")
            logger.info(f"Tamanho do contexto: {len(context)} caracteres")
            response = self.ask_question_with_context(question, context, model_name)
            
            # Verifica se a resposta indica que o conhecimento geral foi necessário
            if "conhecimento geral" in response.lower():
                logger.info("\nResposta não encontrada no contexto. Usando conhecimento base do modelo...")
                general_response = self.ask_question_general(question, model_name)
                
                # Combina as respostas
                final_response = f"""Baseado nos documentos fornecidos: {response}

Complementando com o conhecimento base do modelo: {general_response}"""
                
                processing_time = time.time() - start_time
                logger.info(f"Resposta gerada em {processing_time:.2f} segundos")
                return final_response
            
            processing_time = time.time() - start_time
            logger.info(f"Resposta gerada em {processing_time:.2f} segundos")
            return response
        else:
            logger.info("\nNenhum contexto fornecido. Usando apenas conhecimento base do modelo...")
            response = self.ask_question_general(question, model_name)
            
            processing_time = time.time() - start_time
            logger.info(f"Resposta gerada em {processing_time:.2f} segundos")
            return response

    def train_model(self, documents: List[str], model_name: str, context: Optional[str] = None) -> str:
        """
        Prepara o modelo para usar tanto os documentos fornecidos quanto seu conhecimento base.
        """
        logger.info(f"\nPreparando modelo {model_name} com documentos personalizados")
        logger.info(f"Número de documentos: {len(documents)}")
        start_time = time.time()
        
        # Processa documentos com barra de progresso
        with tqdm(total=len(documents), desc="Processando documentos") as pbar:
            context_text = ""
            for doc in documents:
                context_text += doc + "\n\n"
                pbar.update(1)
        
        logger.info(f"Tamanho total do contexto: {len(context_text)} caracteres")
        
        system_prompt = """Você é um assistente especializado que combina conhecimento dos documentos 
        fornecidos com seu conhecimento base. Ao responder perguntas, primeiro procure nos documentos 
        e, se necessário, complemente com seu conhecimento geral."""
        
        training_prompt = f"""Analise os seguintes documentos que serão usados como fonte primária 
        de informações:

        {context_text}

        Instruções:
        1. Use estes documentos como fonte principal de informações
        2. Quando necessário, complemente com seu conhecimento base
        3. Sempre indique explicitamente quando estiver usando cada fonte

        Confirme que você está pronto para:
        - Primeiro buscar respostas nos documentos fornecidos
        - Complementar com seu conhecimento base quando necessário"""
        
        logger.info("\nEnviando documentos para o modelo...")
        response = self.process_with_model(
            text=training_prompt,
            model_name=model_name,
            system_prompt=system_prompt
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Treinamento concluído em {processing_time:.2f} segundos")
        logger.info("\nResposta do modelo:")
        logger.info(response)
        return response

# Inicializa a API do Ollama
ollama_api = OllamaAPI()
