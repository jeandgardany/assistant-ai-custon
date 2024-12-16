from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

from url_processor import URLProcessor
from vectorizer import OllamaAPI
from database import (
    ensure_database_exists,
    save_to_database,
    get_saved_data,
    get_relevant_chunks
)

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuração do Flask
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit
app.config['UPLOAD_FOLDER'] = 'uploads'

# Criar pasta de uploads se não existir
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Inicialização das APIs e processadores
url_processor = URLProcessor()
ollama_api = OllamaAPI()

# Garantir que o banco de dados existe
ensure_database_exists()

def process_file(file_path: str, file_type: str) -> Optional[str]:
    """Processa diferentes tipos de arquivos."""
    try:
        if file_type == 'application/pdf' or file_path.lower().endswith('.pdf'):
            import PyPDF2
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                content = []
                total_pages = len(pdf_reader.pages)
                logger.info(f"Processando PDF com {total_pages} páginas")
                
                for i, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():  # Só adiciona se tiver texto
                        content.append(text)
                    logger.info(f"Página {i+1}/{total_pages} processada")
                
                full_content = "\n\n".join(content)
                logger.info(f"PDF processado: {len(full_content)} caracteres")
                return full_content
                
        elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' or file_path.lower().endswith('.docx'):
            from docx import Document
            doc = Document(file_path)
            content = []
            
            logger.info("Processando documento DOCX")
            for para in doc.paragraphs:
                if para.text.strip():  # Só adiciona se tiver texto
                    content.append(para.text)
            
            full_content = "\n\n".join(content)
            logger.info(f"DOCX processado: {len(full_content)} caracteres")
            return full_content
            
        else:
            # Para arquivos de texto
            logger.info("Processando arquivo de texto")
            encodings = ['utf-8', 'latin1', 'cp1252']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                        logger.info(f"Arquivo de texto processado: {len(content)} caracteres")
                        return content
                except UnicodeDecodeError:
                    continue
            
            raise ValueError(f"Não foi possível decodificar o arquivo com as codificações: {encodings}")
            
    except Exception as e:
        logger.error(f"Erro ao processar arquivo {file_path}: {str(e)}")
        return None

@app.route('/')
def index():
    """Página principal do dashboard."""
    models = ollama_api.list_models()
    return render_template('index.html', models=models)

@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():
    """Rota para upload de arquivos."""
    if request.method == 'GET':
        models = ollama_api.list_models()
        return render_template('upload_file.html', models=models)
        
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'Nenhum arquivo enviado'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Nenhum arquivo selecionado'}), 400
        
        if file:
            if not file.filename:
                return jsonify({'error': 'Nome do arquivo inválido'}), 400
                
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            try:
                content = process_file(file_path, file.content_type or 'text/plain')
                if content:
                    # Log do form data recebido
                    logger.info("Form data recebido:")
                    for key in request.form:
                        logger.info(f"- {key}: {request.form[key]}")
                    
                    # Processa o conteúdo
                    chunks_data = process_content(content, url_processor)
                    logger.info(f"Chunks criados: {len(chunks_data)}")
                    
                    # Obtém o model_name do form
                    model_name = request.form.get('model_name')
                    logger.info(f"Model name recebido: {model_name}")
                    
                    # Salva no banco de dados
                    save_to_database(
                        content=content,
                        model_name=model_name if model_name else 'mistral',
                        source_type='file',
                        source_path=filename,
                        chunks_data=chunks_data
                    )
                    
                    logger.info("Documento salvo no banco de dados")
                    
                    return jsonify({'message': 'Arquivo processado com sucesso!'})
                else:
                    return jsonify({'error': 'Erro ao processar arquivo'}), 400
                    
            except Exception as e:
                logger.error(f"Erro no processamento do arquivo: {str(e)}")
                return jsonify({'error': str(e)}), 500
            finally:
                # Limpa o arquivo após processamento
                if os.path.exists(file_path):
                    os.remove(file_path)
    
    # Se chegou aqui com POST, retorna com os modelos
    models = ollama_api.list_models()
    return render_template('upload_file.html', models=models)

@app.route('/upload_data', methods=['GET', 'POST'])
def upload_data():
    """Rota para upload de dados via JSON."""
    if request.method == 'GET':
        models = ollama_api.list_models()
        return render_template('upload_data.html', models=models)
        
    if request.method == 'POST':
        try:
            data = request.get_json()
            if not data or 'content' not in data:
                return jsonify({'error': 'Dados inválidos'}), 400
            
            content = data['content']
            model_name = data.get('model_name', 'mistral')
            
            # Se o conteúdo é uma URL, processa ela primeiro
            if content.startswith('http://') or content.startswith('https://'):
                logger.info(f"Processando URL: {content}")
                content = url_processor.extract_content_from_url(content)
                if not content:
                    return jsonify({'error': 'Não foi possível extrair conteúdo da URL'}), 400
                source_type = 'url'
                source_path = data['content']  # URL original
            else:
                source_type = 'text'
                source_path = None
            
            # Processa o conteúdo
            chunks_data = process_content(content, url_processor)
            logger.info(f"Chunks criados: {len(chunks_data)}")
            
            # Salva no banco de dados
            save_to_database(
                content=content,
                model_name=model_name,
                source_type=source_type,
                source_path=source_path,
                chunks_data=chunks_data
            )
            
            return jsonify({'message': 'Dados processados com sucesso!'})
            
        except Exception as e:
            logger.error(f"Erro no processamento dos dados: {str(e)}")
            return jsonify({'error': str(e)}), 500

@app.route('/files')
def list_files():
    """Lista todos os documentos salvos."""
    try:
        documents = get_saved_data()
        models = ollama_api.list_models()
        return render_template('list_files.html', saved_data=documents, models=models)
    except Exception as e:
        logger.error(f"Erro ao listar arquivos: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train_model():
    """Treina o modelo com os documentos salvos."""
    try:
        data = request.get_json()
        model_name = data.get('model_name', 'mistral')
        
        # Recupera todos os documentos
        documents = get_saved_data()
        if not documents:
            return jsonify({'error': 'Nenhum documento encontrado para treinamento'}), 400
        
        # Prepara os documentos para treinamento
        doc_contents = [doc['content'] for doc in documents]
        
        # Treina o modelo
        response = ollama_api.train_model(doc_contents, model_name)
        
        return jsonify({
            'message': 'Modelo treinado com sucesso!',
            'details': response
        })
        
    except Exception as e:
        logger.error(f"Erro no treinamento do modelo: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    """Processa perguntas usando o modelo treinado."""
    try:
        data = request.get_json()
        question = data.get('question')
        model_name = data.get('model_name', 'mistral')
        
        if not question:
            return jsonify({'error': 'Pergunta não fornecida'}), 400
        
        # Recupera chunks relevantes
        relevant_chunks = get_relevant_chunks(question)
        context = "\n".join([chunk[0] for chunk in relevant_chunks]) if relevant_chunks else None
        
        # Processa a pergunta
        answer = ollama_api.ask_question(question, model_name, context)
        
        return jsonify({'answer': answer})
        
    except Exception as e:
        logger.error(f"Erro ao processar pergunta: {str(e)}")
        return jsonify({'error': str(e)}), 500

def process_content(content: str, processor: URLProcessor) -> List[Dict[str, Any]]:
    """Processa o conteúdo usando o URLProcessor."""
    chunks = processor.create_chunks(content)
    vectors, scores = processor.vectorize_chunks(chunks)
    
    chunks_data = []
    for i, (chunk, vector, score) in enumerate(zip(chunks, vectors, scores)):
        chunks_data.append({
            'content': chunk,
            'vector': vector.tolist(),
            'score': float(score),
            'chunk_size': processor.chunk_size,
            'overlap': processor.overlap,
            'index': i
        })
    
    return chunks_data

if __name__ == '__main__':
    app.run(debug=True, port=5000)
