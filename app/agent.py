import os
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
from dotenv import load_dotenv

def setup_compliance_qa_chain(
    collection_name: str = "islamic_finance_guidelines",
    persist_directory: str = "../chroma_db"
):
    
    load_dotenv()
    # Set Groq API key
    os.environ["GROQ_API_KEY"] =  os.getenv("GROQ_API_KEY")

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=persist_directory)

    # Create a collection if it doesn't exist
    try:
        # Check if collection exists by attempting to get it
        collection = client.get_collection(name=collection_name)
        print(f"Using existing collection: {collection_name}")
    except Exception as e:
        print(f"Collection doesn't exist or error occurred: {e}")
        print(f"Creating new collection: {collection_name}")
        # Let Langchain create the collection when initializing Chroma

    # Initialize Chroma vector store using Langchain's integration
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )

    # Set up the retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # Create the prompt
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a legal expert specializing in Islamic finance and general compliance. You will assess whether the user's query complies with the relevant regulations or not.

Provide your response strictly in the following JSON format:
{{
  "answer": "yes" or "no",
  "reason": "a concise paragraph explaining why the answer is yes or no",
  "score": "green" (compliant), "yellow" (uncertain or partially compliant), or "red" (non-compliant),
  "alternative": "if the answer is no, propose a better solution"
}}

User Query:
{question}

Relevant Law Excerpts:
{context}

Now respond with the JSON:
"""
    )

    # Initialize Groq LLM
    llm = ChatGroq(
        model_name="llama3-70b-8192",  # or other models like "mixtral-8x7b-32768"
        temperature=0,
        max_tokens=1024
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return qa_chain, vectorstore

def load_PDF_doc(pdf_directory: str):
    documents = []

    if not os.path.exists(pdf_directory):
        print(f"Directory {pdf_directory} does not exist.")
    print(f"Loading PDF: {pdf_directory}")

    # Use PyPDFLoader to extract text from the PDF
    loader = PyPDFLoader(pdf_directory)
    pdf_documents = loader.load()

    # Add source information to metadata
    for doc in pdf_documents:
        doc.metadata["source"] = pdf_directory
        doc.metadata["filename"] = pdf_directory

    documents.extend(pdf_documents)
    print(f"Extracted {len(pdf_documents)} pages from {pdf_directory}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    chunked_documents = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunked_documents)} chunks")
    query_text = "\n".join([chunk.page_content for chunk in chunked_documents])
    return query_text

def run_compliance_query(query: str, qa_chain, output_path: str = "compliance_result.json"):
    result = qa_chain.invoke({"query": query})
    model_output = result["result"]

    try:
        output_json = json.loads(model_output)
    except json.JSONDecodeError as e:
        print("Failed to parse JSON:", e)
        output_json = {"error": "Model returned invalid JSON", "raw": model_output}

    with open(output_path, "w") as f:
        json.dump(output_json, f, indent=4)

    print(f"Saved result to {output_path}")
    return output_json


if __name__ == "__main__":
    qa_chain, vectorstore = setup_compliance_qa_chain()
    query = load_PDF_doc("./tests\MODEL MURABAHA FACILITY AGREEMENT.pdf")
    query = query[:8000]
    response = run_compliance_query(query, qa_chain)
