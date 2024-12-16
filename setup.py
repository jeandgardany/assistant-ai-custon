import subprocess
import sys
import os
from typing import List, Tuple
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_python_version() -> bool:
    """Verifica se a versão do Python é compatível."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error("Python 3.8 ou superior é necessário")
        return False
    return True

def check_ollama() -> bool:
    """Verifica se o Ollama está instalado e rodando."""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        logger.error("Ollama não encontrado. Por favor, instale o Ollama primeiro.")
        logger.info("Instruções: https://ollama.ai/download")
        return False

def install_requirements() -> bool:
    """Instala as dependências do projeto."""
    try:
        logger.info("Instalando dependências...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Erro ao instalar dependências: {e}")
        return False

def setup_database() -> bool:
    """Configura o banco de dados."""
    try:
        logger.info("Configurando banco de dados...")
        import database
        database.ensure_database_exists()
        return True
    except Exception as e:
        logger.error(f"Erro ao configurar banco de dados: {e}")
        return False

def check_gpu() -> Tuple[bool, str]:
    """Verifica se há suporte a GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return True, f"GPU disponível: {torch.cuda.get_device_name(0)}"
        return False, "GPU não disponível"
    except ImportError:
        return False, "PyTorch não instalado"

def pull_mistral_model() -> bool:
    """Baixa o modelo Mistral do Ollama."""
    try:
        logger.info("Baixando modelo Mistral...")
        result = subprocess.run(['ollama', 'pull', 'mistral'], check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        logger.error(f"Erro ao baixar modelo Mistral: {e}")
        return False

def create_env_file() -> bool:
    """Cria arquivo .env se não existir."""
    if not os.path.exists('.env'):
        try:
            with open('.env', 'w') as f:
                f.write("""# Configurações de Chunks
CHUNK_SIZE=1000     # Tamanho máximo de cada chunk em caracteres
CHUNK_OVERLAP=100   # Quantidade de caracteres de sobreposição entre chunks

# Configurações do Ollama
OLLAMA_BASE_URL=http://localhost:11434

# Configurações de Logging
LOG_LEVEL=INFO

# Configurações de GPU
USE_GPU=true        # Tentar usar GPU se disponível
GPU_BATCH_SIZE=32   # Tamanho do batch para processamento em GPU

# Configurações do Banco de Dados
DATABASE_FILE=data.db

# Configurações de Processamento
MAX_WORKERS=4       # Número máximo de workers para processamento paralelo
TIMEOUT=30         # Timeout em segundos para requisições""")
            return True
        except Exception as e:
            logger.error(f"Erro ao criar arquivo .env: {e}")
            return False
    return True

def main():
    """Função principal de setup."""
    logger.info("Iniciando setup do Agente IA...")
    
    # Lista de verificações
    checks: List[Tuple[str, callable]] = [
        ("Versão do Python", check_python_version),
        ("Ollama instalado", check_ollama),
        ("Dependências", install_requirements),
        ("Banco de dados", setup_database),
        ("Arquivo .env", create_env_file),
        ("Modelo Mistral", pull_mistral_model)
    ]
    
    # Executa verificações
    all_passed = True
    for name, check in checks:
        logger.info(f"Verificando {name}...")
        if not check():
            all_passed = False
            logger.error(f"Falha na verificação: {name}")
            break
    
    # Verifica GPU
    has_gpu, gpu_info = check_gpu()
    logger.info(f"Status GPU: {gpu_info}")
    
    if all_passed:
        logger.info("""
Setup concluído com sucesso!

Para iniciar o sistema:
1. Certifique-se de que o Ollama está rodando:
   ollama serve
   
2. Inicie a aplicação:
   python app.py
   
3. Acesse no navegador:
   http://localhost:5000
""")
    else:
        logger.error("Setup não foi concluído devido a erros.")

if __name__ == "__main__":
    main()
