import os
import sys
import logging
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

# API Key Setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

# You can also assign your OpenAI API Key directly here instead of setting it as an environment variable
# OPENAI_API_KEY = "your_openai_api_key"

# Logger Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reusable Embeddings Initialization
EMBEDDINGS = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5", encode_kwargs={"normalize_embeddings": True})

# Default Template for Chat Prompt
default_template = """
You are a helpful assistant with access to specific documents.
Use the information from these documents to answer the following question:
<context>
{context}
</context>
Question: {input}
"""

def get_vectorstore_path(pdf_name):
    return f"vectorstore_{pdf_name}.db"

def get_vectorstore(pdf_file, force_reload=False, batch_size=100):
    pdf_name = Path(pdf_file).stem
    vectorstore_path = get_vectorstore_path(pdf_name)

    if os.path.exists(vectorstore_path) and not force_reload:
        return FAISS.load_local(vectorstore_path, EMBEDDINGS, allow_dangerous_deserialization=True)

    # Load and Process PDF
    loader = PyPDFLoader(pdf_file)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    text_chunks = text_splitter.split_documents(documents)
    
    # Initialize Vectorstore with Batches
    vectorstore = FAISS.from_documents(text_chunks, EMBEDDINGS)
    vectorstore.save_local(vectorstore_path)
    logger.info(f"Vectorstore saved at {vectorstore_path}")

    return vectorstore

def list_vectorstores():
    return [f.replace("vectorstore_", "").replace(".db", "") for f in os.listdir() if f.startswith("vectorstore_") and f.endswith(".db")]

def create_chat_chain(vectorstore, template=default_template):
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)
    prompt_template = ChatPromptTemplate.from_template(template)
    doc_chain = create_stuff_documents_chain(llm, prompt_template)
    return create_retrieval_chain(retriever, doc_chain)

def upload_pdf(file_path):
    if file_path.endswith('.pdf'):
        try:
            get_vectorstore(file_path, force_reload=True)
            print(f"Vector store created/updated for {file_path}.")
        except Exception as e:
            logger.error(f"Error uploading PDF: {e}")
            print(f"Error: {e}")
    else:
        print("Only PDF files are supported.")

def chat(pdf_name, custom_template=default_template):
    vectorstore_path = get_vectorstore_path(pdf_name)
    if not os.path.exists(vectorstore_path):
        print("Vector store not found. Please upload the PDF first.")
        return

    vectorstore = get_vectorstore(f"{pdf_name}.pdf")
    chain = create_chat_chain(vectorstore, custom_template)
    
    print("\nEnter 'exit' to return to the main menu.")
    while True:
        user_input = input("Enter your question: ")
        if user_input.lower() == 'exit':
            break

        response = chain.invoke({"input": user_input})
        print("\nResponse:", response['answer'])

def main_menu():
    while True:
        print("\n--- PDF Chatbot ---")
        print("1. Upload PDF")
        print("2. List available PDFs")
        print("3. Chat with a PDF")
        print("4. Exit")
        choice = input("Enter your choice (1-4): ")

        if choice == '1':
            file_path = input("Enter the path to the PDF file: ")
            upload_pdf(file_path)
        elif choice == '2':
            pdfs = list_vectorstores()
            print("\nAvailable PDFs:")
            for pdf in pdfs:
                print(f"- {pdf}")
        elif choice == '3':
            pdfs = list_vectorstores()
            if not pdfs:
                print("No PDFs available. Please upload a PDF first.")
                continue
            print("\nAvailable PDFs:")
            for i, pdf in enumerate(pdfs, 1):
                print(f"{i}. {pdf}")
            pdf_choice = int(input("Choose a PDF (enter the number): ")) - 1
            if 0 <= pdf_choice < len(pdfs):
                pdf_name = pdfs[pdf_choice]
                question = input("Enter your question: ")
                chat(pdf_name, question)
            else:
                print("Invalid choice.")
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()