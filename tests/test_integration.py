import unittest
import os
import sys
from pathlib import Path

# Adiciona o diretório raiz ao PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

from vectorizer import OllamaAPI
from url_processor import URLProcessor
from database import ensure_database_exists, save_to_database, get_saved_data

class TestIntegration(unittest.TestCase):
    """Testes de integração do sistema."""
    
    @classmethod
    def setUpClass(cls):
        """Configuração inicial dos testes."""
        cls.ollama_api = OllamaAPI()
        cls.url_processor = URLProcessor()
        ensure_database_exists()
        
        # Cria arquivo de teste
        cls.test_content = """
        Este é um texto de teste para verificar o funcionamento do sistema.
        O agente IA deve ser capaz de processar este texto e responder perguntas sobre ele.
        O texto contém informações sobre teste e processamento de documentos.
        """
        
        with open("test_document.txt", "w", encoding="utf-8") as f:
            f.write(cls.test_content)
    
    def test_1_ollama_connection(self):
        """Testa a conexão com o Ollama."""
        models = self.ollama_api.list_models()
        self.assertIsInstance(models, list)
        self.assertIn("mistral", models)
    
    def test_2_document_processing(self):
        """Testa o processamento de documentos."""
        # Processa o documento de teste
        chunks = self.url_processor.create_chunks(self.test_content)
        vectors, scores = self.url_processor.vectorize_chunks(chunks)
        
        self.assertGreater(len(chunks), 0)
        self.assertEqual(len(chunks), len(vectors))
        self.assertEqual(len(chunks), len(scores))
    
    def test_3_database_operations(self):
        """Testa operações no banco de dados."""
        # Salva documento no banco
        chunks_data = []
        chunks = self.url_processor.create_chunks(self.test_content)
        vectors, scores = self.url_processor.vectorize_chunks(chunks)
        
        for i, (chunk, vector, score) in enumerate(zip(chunks, vectors, scores)):
            chunks_data.append({
                'content': chunk,
                'vector': vector.tolist(),
                'score': float(score),
                'chunk_size': self.url_processor.chunk_size,
                'overlap': self.url_processor.overlap,
                'index': i
            })
        
        save_to_database(
            content=self.test_content,
            model_name='mistral',
            source_type='test',
            source_path='test_document.txt',
            chunks_data=chunks_data,
            url_processor=self.url_processor
        )
        
        # Verifica se os dados foram salvos
        saved_data = get_saved_data()
        self.assertGreater(len(saved_data), 0)
    
    def test_4_question_answering(self):
        """Testa o sistema de perguntas e respostas."""
        question = "Sobre o que é este texto?"
        response = self.ollama_api.ask_question(
            question=question,
            model_name="mistral",
            context=self.test_content
        )
        
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
    
    @classmethod
    def tearDownClass(cls):
        """Limpeza após os testes."""
        # Remove arquivo de teste
        if os.path.exists("test_document.txt"):
            os.remove("test_document.txt")

if __name__ == '__main__':
    unittest.main(verbosity=2)
