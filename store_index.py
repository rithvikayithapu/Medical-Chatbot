from src.helper import load_pdf, text_split, download_huggingface_embeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_huggingface_embeddings()

index_name="medical-chatbot"

# Create embedding for each of the text chunks and store them in Pinecone
docsearch = PineconeVectorStore.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)