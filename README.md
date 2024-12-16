# Agente IA com Ollama Mistral 7B

Este é um sistema de agente IA que permite processar e armazenar documentos e URLs em um banco de dados vetorizado, utilizando o modelo Mistral 7B através do Ollama para processamento de linguagem natural.

## Requisitos

- Python 3.8+
- Ollama instalado e configurado com o modelo Mistral 7B
- CUDA (opcional, para aceleração GPU)

## Instalação

1. Clone o repositório:
```bash
git clone <seu-repositorio>
cd <pasta-do-projeto>
```

2. Instale o Ollama (se ainda não tiver instalado):
- Windows: [Instruções de instalação do Ollama](https://ollama.ai/download)
- Baixe o modelo Mistral:
```bash
ollama pull mistral
```

3. Instale as dependências Python:
```bash
pip install -r requirements.txt
```

4. Configure o arquivo .env:
```bash
cp .env.example .env
```

## Uso

1. Inicie o servidor Ollama:
```bash
ollama serve
```

2. Inicie a aplicação:
```bash
python app.py
```

3. Acesse a interface web:
- Abra o navegador em `http://localhost:5000`

## Funcionalidades

- Upload de documentos (PDF, DOCX, TXT)
- Processamento de URLs
- Vetorização de texto com chunks otimizados
- Interface web intuitiva
- Suporte a GPU para processamento acelerado
- Armazenamento em banco de dados SQLite
- Perguntas e respostas contextualizadas

## Configuração

O arquivo `.env` permite configurar:

- Tamanho dos chunks de texto
- Sobreposição entre chunks
- URL base do Ollama
- Configurações de GPU
- Parâmetros de logging
- Configurações do banco de dados

## Processamento de Documentos

O sistema processa documentos da seguinte forma:

1. Extração de texto do documento/URL
2. Divisão em chunks com sobreposição
3. Vetorização usando TF-IDF
4. Armazenamento no banco de dados
5. Indexação para busca rápida

## API

O sistema expõe as seguintes rotas:

- `/upload`: Upload de arquivos
- `/upload_data`: Upload de texto ou URL
- `/ask`: Fazer perguntas ao modelo
- `/train`: Treinar modelo com documentos
- `/files`: Listar documentos salvos

## Contribuição

Contribuições são bem-vindas! Por favor, siga estas etapas:

1. Fork o repositório
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## Licença

MIT License
