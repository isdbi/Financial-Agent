# Run this script to insert data from the "data" folder into the vector database

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"

import json
import uuid
from langchain_groq import ChatGroq
from langchain_community.embeddings import SentenceTransformerEmbeddings

from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import chromadb
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_pdf_documents(pdf_directory):
    """
    Load documents from PDF files in a directory.
    Returns a list of Document objects suitable for adding to the vector store.

    Args:
        pdf_directory (str): Path to directory containing PDF files

    Returns:
        list: List of Document objects with text content and metadata
    """
    documents = []

    if not os.path.exists(pdf_directory):
        print(f"Directory {pdf_directory} does not exist.")
        return documents

    for filename in os.listdir(pdf_directory):
        if filename.lower().endswith('.pdf'):
            try:
                file_path = os.path.join(pdf_directory, filename)
                print(f"Loading PDF: {file_path}")

                # Use PyPDFLoader to extract text from the PDF
                loader = PyPDFLoader(file_path)
                pdf_documents = loader.load()

                # Add source information to metadata
                for doc in pdf_documents:
                    doc.metadata["source"] = file_path
                    doc.metadata["filename"] = filename

                documents.extend(pdf_documents)
                print(f"Extracted {len(pdf_documents)} pages from {filename}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into smaller chunks for better retrieval.

    Args:
        documents (list): List of Document objects
        chunk_size (int): Maximum size of each text chunk
        chunk_overlap (int): Overlap between consecutive chunks

    Returns:
        list: List of chunked Document objects
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunked_documents = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunked_documents)} chunks")

    return chunked_documents


def load_documents_from_directory(directory_path):
    """
    Load documents from text files in a directory.
    Returns a list of Document objects suitable for adding to the vector store.
    """
    from langchain.docstore.document import Document
    documents = []

    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist.")
        return documents

    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                doc = Document(
                    page_content=content,
                    metadata={"source": file_path}
                )
                documents.append(doc)

    return documents

def add_documents_to_vectorstore(vectorstore, documents):
    """Add documents to the vector store."""
    vectorstore.add_documents(documents)
    vectorstore.persist()
    print(f"Added {len(documents)} documents to the vector store")


def upsert_pdfs_to_vectordb(pdf_directory, collection_name="islamic_finance_guidelines", persist_directory="../chroma_db", chunk_size=1000, chunk_overlap=200):
    """
    Load PDFs from a directory, split them into chunks, and upsert them to the ChromaDB vector database.

    Args:
        pdf_directory (str): Path to directory containing PDF files
        collection_name (str): Name of the ChromaDB collection
        persist_directory (str): Directory to persist the ChromaDB data
        chunk_size (int): Maximum size of each text chunk
        chunk_overlap (int): Overlap between consecutive chunks

    Returns:
        tuple: (qa_chain, vectorstore) - The QA chain and vector store objects
    """
    # Load the PDFs
    print(f"Loading PDFs from {pdf_directory}...")
    documents = load_pdf_documents(pdf_directory)

    if not documents:
        print("No PDF documents found or loaded.")
        return None, None

    # Split documents into chunks
    print("Splitting documents into chunks...")
    chunked_docs = split_documents(documents, chunk_size, chunk_overlap)

    # Initialize embeddings
    print("Setting up embeddings and vector store...")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create a new vectorstore with the documents
    print(f"Creating vector store with {len(chunked_docs)} document chunks...")
    vectorstore = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory
    )

    vectorstore.persist()
    print(f"Successfully added {len(chunked_docs)} document chunks to ChromaDB collection '{collection_name}'")

if __name__ == "__main__":
    pdf_directory = "./data"
    upsert_pdfs_to_vectordb(pdf_directory)
    print("Data insertion complete.")