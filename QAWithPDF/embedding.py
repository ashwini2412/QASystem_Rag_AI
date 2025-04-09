import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.gemini import GeminiEmbedding
from data_ingestion import load_data
from model_api import load_model
import sys
from custom_exception import customexception
from logger import logging

load_dotenv()

def download_gemini_embedding(model, documents):
    """
    Downloads and initializes a Gemini Embedding model for vector embeddings.
    
    Args:
        model: The LLM model to use
        documents: List of documents to index
        
    Returns:
        query_engine: Configured query engine for the indexed documents
    """
    try:
        logging.info("Initializing Gemini embedding model")
        
        # Initialize embedding model
        gemini_embed_model = GeminiEmbedding(
            model_name="models/embedding-001",
            api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Configure global settings
        Settings.embed_model = gemini_embed_model
        Settings.llm = model
        Settings.node_parser = SentenceSplitter(
            chunk_size=800, 
            chunk_overlap=20
        )
        
        logging.info("Creating vector store index")
        
        # Create index
        index = VectorStoreIndex.from_documents(
            documents,
            transformations=[SentenceSplitter(chunk_size=800, chunk_overlap=20)]
        )
        
        logging.info("Persisting storage context")
        index.storage_context.persist()
        
        logging.info("Creating query engine")
        query_engine = index.as_query_engine()
        
        return query_engine
        
    except Exception as e:
        logging.error(f"Error in download_gemini_embedding: {str(e)}")
        raise customexception(e, sys)