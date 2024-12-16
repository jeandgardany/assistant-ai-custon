import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from typing import List, Dict, Optional, Any, Union, Tuple
import re
from tqdm import tqdm
import logging
import time
import os
from dotenv import load_dotenv
from scipy.sparse import spmatrix, csr_matrix

# Carrega variáveis de ambiente
load_dotenv()

# Configurar logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Variáveis globais
cp: Any = None
USE_GPU: bool = os.getenv('USE_GPU', 'false').lower() == 'true'

def init_gpu():
    """Inicializa o suporte à GPU."""
    global cp, USE_GPU
    try:
        import cupy
        cp = cupy
        USE_GPU = True
        logger.info("GPU disponível! Usando CuPy para aceleração.")
    except ImportError:
        logger.info("GPU não disponível. Usando NumPy para CPU.")

# Inicializa o suporte à GPU
init_gpu()

class URLProcessor:
    def __init__(self, chunk_size: Optional[int] = None, overlap: Optional[int] = None):
        """
        Inicializa o processador de URLs.
        
        Args:
            chunk_size: Tamanho máximo de cada chunk em caracteres
            overlap: Quantidade de caracteres de sobreposição entre chunks
        """
        try:
            # Tenta carregar do .env ou usa valores padrão
            env_chunk_size = os.getenv('CHUNK_SIZE')
            env_overlap = os.getenv('CHUNK_OVERLAP')
            
            self.chunk_size = chunk_size or (int(env_chunk_size) if env_chunk_size else 1000)
            self.overlap = overlap or (int(env_overlap) if env_overlap else 100)
            self.vectorizer = TfidfVectorizer()
            self.use_gpu = os.getenv('USE_GPU', 'false').lower() == 'true'
            
            logger.info(f"Inicializando URLProcessor com configurações:")
            logger.info(f"- chunk_size: {self.chunk_size}")
            logger.info(f"- overlap: {self.overlap}")
            logger.info(f"- use_gpu: {self.use_gpu}")
        except ValueError as e:
            logger.warning(f"Erro ao carregar configurações do .env: {str(e)}")
            logger.info("Usando valores padrão")
            self.chunk_size = chunk_size or 1000
            self.overlap = overlap or 100
            self.vectorizer = TfidfVectorizer()
            self.use_gpu = False
        
    def extract_content_from_url(self, url: str) -> str:
        """Extrai o conteúdo textual de uma URL."""
        try:
            logger.info(f"Extraindo conteúdo da URL: {url}")
            start_time = time.time()
            
            response = requests.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove scripts, styles e tags de navegação
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'iframe']):
                element.decompose()
            
            # Extrai o texto principal
            text = soup.get_text()
            
            # Limpa o texto
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Remove múltiplos espaços em branco
            text = re.sub(r'\s+', ' ', text).strip()
            
            processing_time = time.time() - start_time
            logger.info(f"Conteúdo extraído: {len(text)} caracteres em {processing_time:.2f} segundos")
            return text
            
        except Exception as e:
            logger.error(f"Erro ao extrair conteúdo da URL {url}: {str(e)}")
            return ""

    def create_chunks(self, text: str) -> List[str]:
        """
        Divide o texto em chunks com sobreposição.
        """
        logger.info("Iniciando criação de chunks")
        logger.info(f"Tamanho do texto a ser dividido: {len(text)} caracteres")
        logger.info(f"Configuração: chunk_size={self.chunk_size}, overlap={self.overlap}")
        
        start_time = time.time()
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0
        
        # Criar barra de progresso
        with tqdm(total=len(words), desc="Criando chunks", unit='words') as pbar:
            for word in words:
                # Adiciona a palavra e um espaço
                word_length = len(word) + 1  # +1 para o espaço
                
                # Se adicionar esta palavra exceder o tamanho máximo do chunk
                if current_length + word_length > self.chunk_size and current_chunk:
                    # Salva o chunk atual
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(chunk_text)
                    
                    # Log a cada 10 chunks
                    if len(chunks) % 10 == 0:
                        logger.info(f"Chunk {len(chunks)} criado: {len(chunk_text)} caracteres")
                    
                    # Mantém as últimas palavras para overlap
                    overlap_words = []
                    overlap_length = 0
                    for w in reversed(current_chunk):
                        if overlap_length + len(w) + 1 <= self.overlap:
                            overlap_words.insert(0, w)
                            overlap_length += len(w) + 1
                        else:
                            break
                    
                    # Inicia novo chunk com as palavras do overlap
                    current_chunk = overlap_words
                    current_length = overlap_length
                
                # Adiciona a palavra ao chunk atual
                current_chunk.append(word)
                current_length += word_length
                pbar.update(1)
            
            # Adiciona o último chunk se houver
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                logger.info(f"Último chunk criado: {len(chunk_text)} caracteres")
        
        processing_time = time.time() - start_time
        logger.info(f"Total de {len(chunks)} chunks criados em {processing_time:.2f} segundos")
        logger.info(f"Tamanho médio dos chunks: {sum(len(c) for c in chunks)/len(chunks):.2f} caracteres")
        return chunks

    def sparse_to_dense(self, sparse_matrix: Union[spmatrix, np.ndarray]) -> np.ndarray:
        """Converte matriz esparsa para densa."""
        if hasattr(sparse_matrix, 'toarray'):
            return sparse_matrix.toarray()
        elif isinstance(sparse_matrix, np.ndarray):
            return sparse_matrix
        return np.array(sparse_matrix)

    def vectorize_chunks(self, chunks: List[str]) -> Tuple[np.ndarray, List[float]]:
        """
        Vetoriza os chunks e calcula scores de relevância usando GPU se disponível.
        """
        if not chunks:
            return np.array([]), []
        
        logger.info("Iniciando vetorização dos chunks")
        start_time = time.time()
        
        # Vetoriza os chunks
        with tqdm(total=len(chunks), desc="Vetorizando chunks", unit='chunk') as pbar:
            # Vetorização inicial
            vectors = self.vectorizer.fit_transform(chunks)
            dense_vectors = self.sparse_to_dense(vectors)
            pbar.update(len(chunks) // 3)
            
            if self.use_gpu and cp is not None:
                try:
                    # Transfere para GPU
                    vectors_gpu = cp.array(dense_vectors)
                    logger.info("Dados transferidos para GPU")
                    pbar.update(len(chunks) // 3)
                    
                    # Calcula scores na GPU
                    scores = []
                    for chunk in chunks:
                        unique_words = len(set(chunk.split()))
                        total_words = len(chunk.split())
                        score = unique_words / total_words if total_words > 0 else 0
                        scores.append(float(score))
                    scores_gpu = cp.array(scores)
                    pbar.update(len(chunks) // 3)
                    
                    # Transfere resultado de volta para CPU
                    vectors_cpu = cp.asnumpy(vectors_gpu)
                    logger.info("Processamento GPU concluído")
                    return vectors_cpu, scores
                except Exception as e:
                    logger.warning(f"Erro ao usar GPU: {str(e)}. Usando CPU.")
                    self.use_gpu = False
            
            # Processamento em CPU
            scores = []
            for i, chunk in enumerate(chunks):
                unique_words = len(set(chunk.split()))
                total_words = len(chunk.split())
                score = unique_words / total_words if total_words > 0 else 0
                scores.append(score)
                if len(chunks) >= 10 and (i + 1) % max(1, len(chunks) // 10) == 0:  # Log a cada 10% se houver 10+ chunks
                    logger.info(f"Processado {i + 1}/{len(chunks)} chunks")
            pbar.update(len(chunks) * 2 // 3)
        
        processing_time = time.time() - start_time
        logger.info(f"Vetorização concluída em {processing_time:.2f} segundos")
        logger.info(f"Usando {'GPU' if self.use_gpu else 'CPU'} para processamento")
        
        return dense_vectors, scores

    def process_url(self, url: str) -> Dict:
        """Processa uma URL completa: extrai, chunka e vetoriza o conteúdo."""
        logger.info(f"\nProcessando URL: {url}")
        start_time = time.time()
        
        # Extrai o conteúdo
        content = self.extract_content_from_url(url)
        if not content:
            return {
                "success": False,
                "error": "Não foi possível extrair conteúdo da URL"
            }
            
        logger.info(f"Conteúdo extraído: {len(content)} caracteres")
        
        # Cria chunks
        chunks = self.create_chunks(content)
        logger.info(f"Total de chunks criados: {len(chunks)}")
        
        # Vetoriza chunks
        vectors, scores = self.vectorize_chunks(chunks)
        
        total_time = time.time() - start_time
        logger.info(f"Processamento total concluído em {total_time:.2f} segundos")
        
        # Prepara resultado
        result = {
            "success": True,
            "url": url,
            "content": content,
            "chunks": chunks,
            "vectors": vectors,
            "scores": scores,
            "metadata": {
                "total_chunks": len(chunks),
                "chunk_size": self.chunk_size,
                "overlap": self.overlap,
                "total_chars": len(content),
                "processing_time": total_time,
                "using_gpu": self.use_gpu,
                "avg_chunk_size": sum(len(c) for c in chunks)/len(chunks)
            }
        }
        
        return result

    def find_most_relevant_chunks(self, query: str, chunks: List[str], vectors: np.ndarray, 
                                top_k: int = 3) -> List[Tuple[str, float]]:
        """Encontra os chunks mais relevantes para uma query."""
        if not chunks or vectors.size == 0:
            return []
            
        logger.info("Buscando chunks relevantes")
        start_time = time.time()
        
        # Vetoriza a query
        query_vector = self.vectorizer.transform([query])
        query_dense = self.sparse_to_dense(query_vector)
        
        if self.use_gpu and cp is not None:
            try:
                # Transfere para GPU
                vectors_gpu = cp.array(vectors)
                query_vector_gpu = cp.array(query_dense)
                logger.info("Dados transferidos para GPU")
                
                # Calcula similaridade na GPU
                similarities = cp.dot(vectors_gpu, query_vector_gpu.T).flatten()
                similarities_cpu = cp.asnumpy(similarities)
                logger.info("Similaridades calculadas na GPU")
            except Exception as e:
                logger.warning(f"Erro ao usar GPU: {str(e)}. Usando CPU.")
                similarities_cpu = np.dot(vectors, query_dense.T).flatten()
        else:
            # Calcula similaridade na CPU
            similarities_cpu = np.dot(vectors, query_dense.T).flatten()
            logger.info("Similaridades calculadas na CPU")
        
        # Encontra os chunks mais relevantes
        top_indices = similarities_cpu.argsort()[-top_k:][::-1]
        
        processing_time = time.time() - start_time
        logger.info(f"Busca concluída em {processing_time:.2f} segundos")
        logger.info(f"Encontrados {top_k} chunks mais relevantes")
        
        return [(chunks[i], float(similarities_cpu[i])) for i in top_indices]
