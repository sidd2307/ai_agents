import os
import tempfile
import re
import pyttsx3  # Added for narration
from datetime import datetime

# To Build UI
import streamlit as st

# For Embedding Model
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Agno Agentic AI Library
from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.vectordb.chroma import ChromaDb

# Langchain for Document Parsing
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- Page Config & Futuristic Styling ---
st.set_page_config(page_title="SiddXDeepSeek", layout="wide")

# Custom CSS for a Cool Futuristic Theme
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #00f2ff;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 12, 41, 0.8) !important;
        border-right: 1px solid #00f2ff;
    }
    
    /* Input Box */
    .stChatInputContainer {
        padding-bottom: 20px;
    }
    
    /* Glassmorphism containers */
    div[data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        border: 1px solid rgba(0, 242, 255, 0.2);
        margin-bottom: 10px;
    }

    /* Titles and Headers */
    h1, h2, h3 {
        color: #00f2ff !important;
        text-shadow: 0 0 10px #00f2ff;
        font-family: 'Courier New', Courier, monospace;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(45deg, #00f2ff, #7000ff);
        color: white;
        border: none;
        border-radius: 20px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        box-shadow: 0 0 15px #00f2ff;
        transform: scale(1.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- Set Google API Key ---
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Missing GOOGLE_API_KEY in secrets!")

# --- Constants ---
COLLECTION_NAME = "deepseek_rag"
EMBEDDING_MODEL = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# --- Session State Initialization ---
session_defaults = {
    "chroma_path": "./chroma_db",
    "model_version": "deepseek-r1:1.5b",
    "vector_store": None,
    "processed_documents": [],
    "history": [],
    "use_web_search": False,
    "force_web_search": False,
    "similarity_threshold": 0.7,
    "rag_enabled": True,
}

for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- App Header ---
st.title("⚡ SiddXDeepSeek")
st.markdown("---")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("🧬 CORE ENGINE")
    st.session_state.model_version = st.radio("Model Version", ["deepseek-r1:1.5b"], help="DeepSeek Reasoning Engine")
    
    st.header("⚙️ RAG SETTINGS")
    st.session_state.rag_enabled = st.toggle("Enable RAG Mode", value=st.session_state.rag_enabled)
    st.session_state.use_web_search = st.checkbox("Web Search Fallback", value=st.session_state.use_web_search)
    
    if st.button("🔥 PURGE HISTORY"):
        st.session_state.history = []
        st.rerun()

# --- Narration Function ---
def narrate_text(text):
    """Narrates the response using pyttsx3."""
    engine = pyttsx3.init()
    # Adjusting rate and volume for a 'cooler' voice feel
    engine.setProperty('rate', 175)
    engine.setProperty('volume', 0.9)
    engine.say(text)
    engine.runAndWait()

# --- Utility Functions (Init, Split, Process) ---
def init_chroma():
    chroma = ChromaDb(
        collection=COLLECTION_NAME,
        path=st.session_state.chroma_path,
        embedder=EMBEDDING_MODEL,
        persistent_client=True
    )
    try:
        chroma.client.get_collection(name=COLLECTION_NAME)
    except Exception:
        chroma.create()
    return chroma

def split_texts(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)
    return [Document(page_content=chunk.page_content, metadata=chunk.metadata) for chunk in split_docs if chunk.page_content.strip()]

def process_pdf(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            loader = PyPDFLoader(tmp_file.name)
            documents = loader.load()
        for doc in documents:
            doc.metadata.update({"source_type": "pdf", "file_name": uploaded_file.name, "timestamp": datetime.now().isoformat()})
        return split_texts(documents)
    except Exception as e:
        st.error(f"❌ PDF Error: {str(e)}")
        return []

def process_web(url: str):
    try:
        loader = WebBaseLoader(url)
        documents = loader.load()
        for doc in documents:
            doc.metadata.update({"source": url, "timestamp": datetime.now().isoformat()})
        return split_texts(documents)
    except Exception as e:
        st.error(f"❌ Web Error: {str(e)}")
        return []

def retrieve_documents(prompt, chroma_client, COLLECTION_NAME):
    vector_store = chroma_client.client.get_collection(name=COLLECTION_NAME)
    results = vector_store.query(query_texts=[prompt], n_results=5)
    docs = results.get('documents', [])
    return docs, len(docs) > 0

def filter_think_tags(response):
    return re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)

# --- Agent Factory ---
def get_web_search_agent():
    return Agent(
        name="Web Search Agent",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGoTools()],
        instructions="Summarize key points from web search.",
        markdown=True,
    )

def get_rag_agent():
    return Agent(
        name="SiddX Agent",
        model=Ollama(id=st.session_state.model_version),
        instructions="Answer using the context. Be concise and technical.",
        markdown=True,
    )

# --- UI Layout: Chat & Controls ---
chat_col, toggle_col = st.columns([0.85, 0.15])
with chat_col:
    prompt = st.chat_input("Enter Command...")
with toggle_col:
    st.session_state.force_web_search = st.toggle('🌐 Force Web', help="Bypass RAG and search web")

# --- File Handling ---
if st.session_state.rag_enabled:
    chroma_client = init_chroma()
    st.sidebar.header("📡 UPLINK DATA")
    uploaded_file = st.sidebar.file_uploader("Upload PDF File", type=["pdf"])
    web_url = st.sidebar.text_input("Enter Data URL")

    if uploaded_file and uploaded_file.name not in st.session_state.processed_documents:
        data = process_pdf(uploaded_file)
        if data:
            collection = chroma_client.client.get_collection(name=COLLECTION_NAME)
            collection.add(
                ids=[f"{uploaded_file.name}_{i}" for i in range(len(data))],
                documents=[doc.page_content for doc in data],
                metadatas=[doc.metadata for doc in data]
            )
            st.session_state.processed_documents.append(uploaded_file.name)
            st.sidebar.success("✅ PDF Encrypted into Vector DB")

    if web_url and web_url not in st.session_state.processed_documents:
        texts = process_web(web_url)
        if texts:
            collection = chroma_client.client.get_collection(name=COLLECTION_NAME)
            collection.add(
                ids=[f"web_{datetime.now().timestamp()}_{i}" for i in range(len(texts))],
                documents=[doc.page_content for doc in texts],
                metadatas=[doc.metadata for doc in texts]
            )
            st.session_state.processed_documents.append(web_url)
            st.sidebar.success("✅ Web Data Synced")

# --- Message Rendering & Processing ---
for i, msg in enumerate(st.session_state.history):
    with st.chat_message(msg["role"]):
        clean_content = filter_think_tags(msg["content"])
        st.write(clean_content)
        if msg["role"] == "assistant":
            if st.button(f"🔊 Narrate", key=f"btn_{i}"):
                narrate_text(clean_content)

if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    context = ""
    if not st.session_state.force_web_search and st.session_state.rag_enabled:
        docs, has_docs = retrieve_documents(prompt, chroma_client, COLLECTION_NAME)
        if has_docs:
            flattened_docs = [paragraph for doc in docs for paragraph in doc]
            context = "\n\n".join(flattened_docs)

    if (st.session_state.force_web_search or not context) and st.session_state.use_web_search:
        with st.spinner("🌐 Accessing Global Grid..."):
            web_results = get_web_search_agent().run(prompt).content
            context = f"Web Search Results:\n{web_results}"

    with st.spinner("🔮 SiddX Reasoning..."):
        full_prompt = f"Context: {context}\n\nQuestion: {prompt}"
        response_obj = get_rag_agent().run(full_prompt)
        response_text = response_obj.content
        
        st.session_state.history.append({"role": "assistant", "content": response_text})
        
        with st.chat_message("assistant"):
            clean_res = filter_think_tags(response_text)
            st.write(clean_res)
            if st.button("🔊 Narrate Response"):
                narrate_text(clean_res)