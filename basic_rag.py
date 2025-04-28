# ===========================================
# STEP 1: Basic RAG System with Gemini + Qdrant
# ===========================================

# --- Imports ---
import os
import tempfile
from datetime import datetime
from typing import List

import streamlit as st
import google.generativeai as genai
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.exa import ExaTools
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# --- Constants ---
COLLECTION_NAME = "basic-gemini-rag1"

# --- Custom Embedder using Gemini API ---
class GeminiEmbedder(Embeddings):
    """Wrapper around Gemini embedding model to conform with LangChain's Embeddings interface."""
    
    def __init__(self, model_name="models/text-embedding-004"):
        if st.session_state.google_api_key:
            os.environ["GOOGLE_API_KEY"] = st.session_state.google_api_key
        genai.configure(api_key=st.session_state.google_api_key)
        self.model = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        response = genai.embed_content(
            model=self.model,
            content=text,
            task_type="retrieval_document"
        )
        return response['embedding']

# --- Streamlit UI ---
st.title("🔍 Basic RAG with Gemini & Qdrant")

# --- Session State Initialization ---
default_keys = {
    'google_api_key': "",
    'qdrant_api_key': "",
    'qdrant_url': "",
    'vector_store': None,
    'processed_documents': [],
    'history': [],
    'similarity_threshold': 0.85
}
for k, v in default_keys.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --- Sidebar for API Key Input ---
st.sidebar.header("API Keys")
st.session_state.google_api_key = st.sidebar.text_input("Google API Key", type="password")
st.session_state.qdrant_api_key = st.sidebar.text_input("Qdrant API Key", type="password")
st.session_state.qdrant_url = st.sidebar.text_input("Qdrant URL", placeholder="https://your-cluster.cloud.qdrant.io:6333")

# --- Sidebar for File Upload ---
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

# --- Qdrant Initialization ---
def init_qdrant():
    """Initialize and return Qdrant client."""
    if not all([st.session_state.qdrant_api_key, st.session_state.qdrant_url]):
        return None
    try:
        return QdrantClient(
            url=st.session_state.qdrant_url,
            api_key=st.session_state.qdrant_api_key,
            timeout=60
        )
    except Exception as e:
        st.error(f"🔴 Qdrant connection failed: {str(e)}")
        return None

# --- PDF Processor ---
def process_pdf(file) -> List:
    """Load and split a PDF file into document chunks."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.getvalue())
            loader = PyPDFLoader(tmp_file.name)
            documents = loader.load()

            # Add metadata to each document
            for doc in documents:
                doc.metadata.update({
                    "source_type": "pdf",
                    "file_name": file.name,
                    "timestamp": datetime.now().isoformat()
                })

            # Split documents into smaller chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            return text_splitter.split_documents(documents)
    except Exception as e:
        st.error(f"PDF processing error: {str(e)}")
        return []

# --- Vector Store Creator ---
def create_vector_store(client, texts):
    """Create a vector store in Qdrant and upload documents."""
    try:
        # Create collection if it doesn't exist
        try:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
        except Exception as e:
            if "already exists" not in str(e).lower():
                raise e

        # Instantiate Qdrant-backed vector store
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=GeminiEmbedder()
        )

        with st.spinner('Uploading documents to Qdrant...'):
            vector_store.add_documents(texts)
            st.success("Documents stored successfully!")
            return vector_store
    except Exception as e:
        st.error(f"Vector store error: {str(e)}")
        return None

# --- RAG Agent ---
def get_rag_agent() -> Agent:
    """Initialize the RAG agent with Gemini."""
    return Agent(
        name="Gemini RAG Agent",
        model=Gemini(id="gemini-2.0-flash-thinking-exp-01-21"),
        instructions="""You are an Intelligent Agent specializing in providing accurate answers.
        
        When given context from documents:
        - Focus on information from the provided documents
        - Be precise and cite specific details
        
        Always maintain high accuracy and clarity in your responses.
        """,
        show_tool_calls=True,
        markdown=True,
    )

# --- File Upload Handling ---
if uploaded_file and uploaded_file.name not in st.session_state.processed_documents:
    with st.spinner("Processing PDF..."):
        texts = process_pdf(uploaded_file)
        client = init_qdrant()
        if not client:
            st.warning("⚠️ Qdrant client could not be initialized. Please check your API key and URL.")
        else:
            st.success("✅ Qdrant client connected!")

        if texts and client:
            if st.session_state.vector_store:
                st.session_state.vector_store.add_documents(texts)
            else:
                st.session_state.vector_store = create_vector_store(client, texts)
            st.session_state.processed_documents.append(uploaded_file.name)

# --- Chat Input & RAG Flow ---
prompt = st.chat_input("Ask a question from your documents")
if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    if st.session_state.vector_store:
        retriever = st.session_state.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 5, "score_threshold": st.session_state.similarity_threshold}
        )
        docs = retriever.invoke(prompt)

        if docs:
            context = "\n\n".join([d.page_content for d in docs])
            st.info(f"📊 Found {len(docs)} relevant documents (similarity > {st.session_state.similarity_threshold})")

            try:
                rag_agent = get_rag_agent()
                full_prompt = f"""Context: {context}

Original Question: {prompt}

Please provide a comprehensive answer based on the available information."""
                response = rag_agent.run(full_prompt)

                st.session_state.history.append({"role": "assistant", "content": response.content})

                with st.chat_message("assistant"):
                    st.write(response.content)

                    # Optional: Expand to show source documents
                    with st.expander("🔍 See document sources"):
                        for i, doc in enumerate(docs, 1):
                            source_type = doc.metadata.get("source_type", "unknown")
                            source_icon = "📄" if source_type == "pdf" else "🌐"
                            source_name = doc.metadata.get("file_name" if source_type == "pdf" else "url", "unknown")
                            st.write(f"{source_icon} Source {i} from {source_name}:")
                            st.write(f"{doc.page_content[:200]}...")

            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")
        else:
            st.warning("No relevant documents found.")
    else:
        st.warning("⚠️ Vector store is not initialized.")

