import subprocess
import sys
import os
import logging
from typing import List, Tuple
import requests
import time

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_ollama_status() -> bool:
    """Verifica se o servidor Ollama está rodando."""
    try:
        response = requests.get('http://localhost:11434/api/tags')
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def wait_for_ollama(timeout: int = 30) -> bool:
    """Aguarda o Ollama iniciar."""
    logger.info("Aguardando Ollama iniciar...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        if check_ollama_status():
            logger.info("Ollama está rodando!")
            return True
        time.sleep(1)
    return False

def run_system_checks() -> List[Tuple[str, bool]]:
    """Executa verificações do sistema."""
    checks = []
    
    # Verifica Python
    python_version = sys.version_info
    checks.append(
        ("Python 3.8+", python_version.major == 3 and python_version.minor >= 8)
    )
    
    # Verifica Ollama
    ollama_running = check_ollama_status()
    checks.append(("Ollama Servidor", ollama_running))
    
    # Verifica banco de dados
    db_exists = os.path.exists('data.db')
    checks.append(("Banco de Dados", db_exists))
    
    # Verifica arquivos necessários
    required_files = [
        'app.py',
        'database.py',
        'vectorizer.py',
        'url_processor.py',
        '.env',
        'requirements.txt'
    ]
    for file in required_files:
        checks.append((f"Arquivo {file}", os.path.exists(file)))
    
    return checks

def print_system_status(checks: List[Tuple[str, bool]]):
    """Imprime o status do sistema."""
    print("\n=== Status do Sistema ===")
    print("------------------------")
    
    all_passed = True
    for name, status in checks:
        status_str = "✓" if status else "✗"
        color = "\033[92m" if status else "\033[91m"  # Verde ou Vermelho
        print(f"{color}{status_str}\033[0m {name}")
        if not status:
            all_passed = False
    
    print("------------------------")
    if all_passed:
        print("\033[92mTodos os checks passaram!\033[0m")
    else:
        print("\033[91mAlguns checks falharam. Verifique os erros acima.\033[0m")
    print()

def run_tests():
    """Executa os testes de integração."""
    logger.info("Iniciando testes de integração...")
    
    # Verifica se Ollama está rodando
    if not check_ollama_status():
        logger.warning("Ollama não está rodando. Tentando iniciar...")
        try:
            subprocess.Popen(['ollama', 'serve'])
            if not wait_for_ollama():
                logger.error("Não foi possível iniciar o Ollama!")
                return False
        except FileNotFoundError:
            logger.error("Ollama não encontrado! Instale o Ollama primeiro.")
            return False
    
    try:
        # Executa os testes
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', 'tests/test_integration.py', '-v'],
            capture_output=True,
            text=True
        )
        
        # Imprime resultado dos testes
        print("\n=== Resultado dos Testes ===")
        print(result.stdout)
        
        if result.returncode != 0:
            print("\n=== Erros ===")
            print(result.stderr)
            return False
            
        return True
    except Exception as e:
        logger.error(f"Erro ao executar testes: {e}")
        return False

def main():
    """Função principal."""
    print("\n=== Verificação do Sistema e Testes ===")
    
    # Executa verificações do sistema
    checks = run_system_checks()
    print_system_status(checks)
    
    # Se todas as verificações passaram, executa os testes
    if all(status for _, status in checks):
        print("Executando testes de integração...")
        if run_tests():
            print("\n\033[92mTodos os testes passaram!\033[0m")
            print("\nO sistema está pronto para uso!")
            print("Para iniciar, execute: python app.py")
        else:
            print("\n\033[91mAlguns testes falharam!\033[0m")
            print("Verifique os erros acima e tente novamente.")
    else:
        print("\nCorreija os problemas acima antes de executar os testes.")

if __name__ == "__main__":
    main()
