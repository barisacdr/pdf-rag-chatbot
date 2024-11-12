# PDF Chatbot using Retrieval-Augmented Generation (RAG)

This project implements a chatbot with Retrieval-Augmented Generation (RAG) to answer questions based on specific PDF documents. Using LangChain and OpenAI's API, the chatbot reads documents, splits text, embeds it using HuggingFace embeddings, stores it in a FAISS vector store, and uses the OpenAI API to generate responses based on the context of your documents.

## Features

- **Upload PDF Documents**: Easily upload PDFs to generate a vector store from each document.
- **Chat with PDFs**: Ask questions related to your uploaded PDFs, and the chatbot will retrieve relevant information to answer based on document context.
- **Persistent Vector Stores**: Vector stores are saved locally for reuse without reprocessing PDFs, improving efficiency.

## Technologies Used

- **LangChain**: Provides document loading, text splitting, and chain-based processing.
- **OpenAI API**: Powers the language model to generate context-aware responses.
- **Hugging Face Embeddings**: Used to create vector representations of text chunks, enhancing document searchability.
- **FAISS (Facebook AI Similarity Search)**: Handles vector storage and efficient similarity search.
- **Python Logging**: Manages logging for clear and organized status updates.
- **pdfplumber**: Processes PDFs for initial document reading (optional but useful for handling complex PDFs).

## Project Structure

- `chatbot.py`: The main script to interact with the chatbot.
- `requirements.txt`: Required libraries.

## Setup

### Prerequisites

1. Python 3.8+
2. A paid OpenAI API Key (set as an environment variable `OPENAI_API_KEY`).  *Note: The OpenAI API key must be associated with a paid account as API usage incurs costs.*

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/barisacdr/pdf-rag-chatbot.git
   cd pdf-rag-chatbot
   ```
2. Install required packages:
    ```bash
   pip install -r requirements.txt
   ```

3. Set your OpenAI API key:
    ```bash
    export OPENAI_API_KEY="your_openai_api_key"
    ```


## Running The Chatbot

Run the chatbot interface:
 ```bash
 python chatbot.py
```

## Usage

### Main Menu

1. Upload PDF: Enter the path to your PDF to create or update its vector store.
2. List Available PDFs: See a list of PDF documents with existing vector stores.
3. Chat with a PDF: Select a PDF and ask questions; the chatbot will retrieve relevant information from the document in a loop, allowing you to ask multiple questions.
4. Exit: Close the application.

### Example Usage

1. Upload a PDF by entering its file path.
2. List available PDFs to verify that the document has been processed.
3. Choose a PDF and start asking questions related to its content.