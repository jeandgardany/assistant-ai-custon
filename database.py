import sqlite3
import json
import os
from url_processor import URLProcessor

def create_connection(db_file):
    """Create a database connection to the SQLite database specified by db_file."""
    conn = sqlite3.connect(db_file)
    return conn

def ensure_database_exists():
    """
    Verifica se o banco de dados existe e está configurado corretamente.
    Se não existir, cria com a estrutura adequada.
    """
    if not os.path.exists('data.db'):
        print("Criando novo banco de dados...")
        initialize_database()
        return
    
    # Verifica se as tabelas necessárias existem
    conn = create_connection('data.db')
    try:
        with conn:
            cursor = conn.cursor()
            
            # Verifica se a tabela chunks tem todas as colunas necessárias
            cursor.execute("PRAGMA table_info(chunks)")
            columns = {row[1] for row in cursor.fetchall()}
            
            # Se faltarem colunas necessárias, faz um backup e recria
            required_columns = {'chunk_size', 'overlap', 'vector', 'relevance_score'}
            if not required_columns.issubset(columns):
                print("Atualizando estrutura do banco de dados...")
                # Faz backup das tabelas existentes
                cursor.execute("ALTER TABLE chunks RENAME TO chunks_old")
                cursor.execute("ALTER TABLE documents RENAME TO documents_old")
                
                # Cria novas tabelas com estrutura atualizada
                initialize_tables(cursor)
                
                # Migra dados das tabelas antigas
                cursor.execute("""
                    INSERT INTO documents (id, content, model_name, source_type, source_path, created_at)
                    SELECT id, content, model_name, source_type, source_path, created_at
                    FROM documents_old
                """)
                
                # Migra chunks com valores padrão para novos campos
                cursor.execute("""
                    INSERT INTO chunks (document_id, content, chunk_index, relevance_score, vector, chunk_size, overlap)
                    SELECT 
                        document_id, content, chunk_index, 
                        COALESCE(relevance_score, 0.5), 
                        COALESCE(vector, '[]'),
                        1000, 100
                    FROM chunks_old
                """)
                
                # Remove tabelas antigas
                cursor.execute("DROP TABLE chunks_old")
                cursor.execute("DROP TABLE documents_old")
                
                print("Migração de dados concluída com sucesso!")
    finally:
        conn.close()

def initialize_tables(cursor):
    """Cria as tabelas do banco de dados."""
    # Tabela principal de documentos
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            content TEXT,
            model_name TEXT DEFAULT 'mistral',
            source_type TEXT,  -- 'file', 'url', ou 'text'
            source_path TEXT,  -- caminho do arquivo ou URL
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Tabela para chunks
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            document_id INTEGER,
            content TEXT,
            chunk_index INTEGER,
            relevance_score REAL,
            vector TEXT,  -- vetor serializado como JSON
            chunk_size INTEGER,  -- tamanho do chunk usado
            overlap INTEGER,     -- sobreposição usada
            FOREIGN KEY (document_id) REFERENCES documents (id)
        )
    """)

def initialize_database():
    """Initialize the database with the correct schema."""
    conn = create_connection('data.db')
    with conn:
        cursor = conn.cursor()
        initialize_tables(cursor)
    conn.close()
    print("Banco de dados inicializado com sucesso!")

def process_content(content, url_processor):
    """
    Processa o conteúdo usando o URLProcessor para garantir consistência.
    """
    chunks = url_processor.create_chunks(content)
    vectors, scores = url_processor.vectorize_chunks(chunks)
    
    chunks_data = []
    for i, (chunk, vector, score) in enumerate(zip(chunks, vectors, scores)):
        chunks_data.append({
            'content': chunk,
            'vector': vector.tolist(),
            'score': score,
            'chunk_size': url_processor.chunk_size,
            'overlap': url_processor.overlap,
            'index': i
        })
    
    return chunks_data

def save_to_database(content, model_name='mistral', source_type='text', source_path=None, chunks_data=None, url_processor=None):
    """
    Salva o documento e seus chunks no banco de dados.
    """
    print("\nIniciando salvamento no banco de dados:")
    print(f"- Model name: {model_name}")
    print(f"- Source type: {source_type}")
    print(f"- Source path: {source_path}")
    print(f"- Tamanho do conteúdo: {len(content)} caracteres")
    
    if url_processor is None:
        print("- Usando URLProcessor padrão")
        url_processor = URLProcessor()
    
    # Se chunks_data não foi fornecido, processa o conteúdo
    if chunks_data is None:
        print("- Processando conteúdo para gerar chunks")
        chunks_data = process_content(content, url_processor)
    
    print(f"- Total de chunks a serem salvos: {len(chunks_data)}")
    
    conn = create_connection('data.db')
    try:
        with conn:
            cursor = conn.cursor()
            
            # Insere o documento principal
            cursor.execute("""
                INSERT INTO documents (content, model_name, source_type, source_path)
                VALUES (?, ?, ?, ?)
            """, (content, model_name, source_type, source_path))
            
            document_id = cursor.lastrowid
            print(f"- Documento inserido com ID: {document_id}")
            
            # Insere os chunks
            for i, chunk_data in enumerate(chunks_data):
                cursor.execute("""
                    INSERT INTO chunks (
                        document_id, content, chunk_index, relevance_score, 
                        vector, chunk_size, overlap
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    document_id,
                    chunk_data['content'],
                    chunk_data['index'],
                    chunk_data['score'],
                    json.dumps(chunk_data['vector']),
                    chunk_data['chunk_size'],
                    chunk_data['overlap']
                ))
                if (i + 1) % 10 == 0:
                    print(f"- {i + 1} chunks salvos...")
            
            print(f"- Todos os {len(chunks_data)} chunks foram salvos com sucesso")
                    
    finally:
        conn.close()

def get_saved_data():
    """Retrieve all saved data from the database."""
    conn = create_connection('data.db')
    try:
        with conn:
            cursor = conn.cursor()
            
            # Primeiro, verifica quantos documentos existem
            cursor.execute("SELECT COUNT(*) FROM documents")
            doc_count = cursor.fetchone()[0]
            print(f"Total de documentos no banco: {doc_count}")
            
            # Busca documentos e seus chunks
            cursor.execute("""
                SELECT 
                    d.id, d.content, d.model_name, d.source_type, d.source_path,
                    c.content as chunk_content, c.relevance_score, c.vector,
                    c.chunk_size, c.overlap
                FROM documents d
                LEFT JOIN chunks c ON d.id = c.document_id
                ORDER BY d.id, c.chunk_index
            """)
            
            rows = cursor.fetchall()
            print(f"Total de linhas retornadas (documentos + chunks): {len(rows)}")
            
            # Organiza os resultados
            documents = {}
            for row in rows:
                doc_id = row[0]
                if doc_id not in documents:
                    documents[doc_id] = {
                        "id": doc_id,
                        "content": row[1],
                        "model_name": row[2],
                        "source_type": row[3],
                        "source_path": row[4],
                        "chunks": []
                    }
                    print(f"Documento {doc_id} encontrado:")
                    print(f"- Model name: {row[2]}")
                    print(f"- Source type: {row[3]}")
                    print(f"- Source path: {row[4]}")
                
                # Adiciona chunk se existir
                if row[5]:  # chunk_content
                    documents[doc_id]["chunks"].append({
                        "content": row[5],
                        "score": row[6],
                        "vector": json.loads(row[7]),
                        "chunk_size": row[8],
                        "overlap": row[9]
                    })
            
            result = list(documents.values())
            print(f"Total de documentos processados: {len(result)}")
            for doc in result:
                print(f"Documento {doc['id']} tem {len(doc['chunks'])} chunks")
            
            return result
    finally:
        conn.close()

def get_relevant_chunks(query, top_k=3):
    """Recupera os chunks mais relevantes para uma query."""
    conn = create_connection('data.db')
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT c.content, c.relevance_score, d.source_path,
                       c.chunk_size, c.overlap
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                ORDER BY c.relevance_score DESC
                LIMIT ?
            """, (top_k,))
            
            return cursor.fetchall()
    finally:
        conn.close()
