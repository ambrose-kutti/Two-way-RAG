import os
import io
import re
import base64
import tempfile
import numpy as np
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import hashlib
from datetime import datetime
from typing import List, Optional
import json
from pathlib import Path

# =========================
# DOCUMENT PROCESSING
# =========================
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import TextLoader, PyPDFLoader
    DOCUMENT_AVAILABLE = True
    from langchain_ollama import OllamaLLM
    llm = OllamaLLM(model="llama3.2:latest", temperature=0)
except ImportError:
    DOCUMENT_AVAILABLE = False
    print(
        "Install: pip install langchain langchain-community langchain-text-splitters faiss-cpu pypdf sentence-transformers"
    )

# =========================
#  SPEECH & TTS
# =========================
try:
    from gtts import gTTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("Install: pip install gtts")

# =========================
#  DOCUMENT STORAGE & PRE-TRAINING
# =========================
DOCUMENTS_DIR = "uploaded_documents"
VECTOR_STORE_DIR = "vector_store"
PRE_TRAINED_DATA_DIR = (
    "pre_trained_data"  # Folder containing your existing TXT files (you can change the name of the folder)
)
# Create directories
os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
os.makedirs(PRE_TRAINED_DATA_DIR, exist_ok=True)

# =========================
#  UTILITY FUNCTIONS
# =========================
def save_uploaded_file(file: UploadFile) -> str:
    """Save uploaded file and return path"""
    file_ext = os.path.splitext(file.filename)[1]
    file_hash = hashlib.md5(file.filename.encode()).hexdigest()[:8]
    filename = f"{file_hash}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{file_ext}"
    file_path = os.path.join(DOCUMENTS_DIR, filename)
    with open(file_path, "wb") as buffer:
        content = file.file.read()
        buffer.write(content)
    return file_path

def load_documents(file_path: str):
    """Load documents based on file type"""
    file_ext = os.path.splitext(file_path)[1].lower()
    try:
        if file_ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        return loader.load()
    except Exception as e:
        print(f"Error loading document: {e}")
        return None

def load_all_training_documents():
    """Load all TXT files from the pre-trained data directory"""
    all_documents = []
    if not os.path.exists(PRE_TRAINED_DATA_DIR):
        print(
            f" Pre-trained data directory '{PRE_TRAINED_DATA_DIR}' does not exist. Creating it."
        )
        os.makedirs(PRE_TRAINED_DATA_DIR, exist_ok=True)
        return all_documents
    # Get all TXT files
    txt_files = [f for f in os.listdir(PRE_TRAINED_DATA_DIR) if f.endswith(".txt")]
    print(f"Found {len(txt_files)} TXT files in '{PRE_TRAINED_DATA_DIR}': {txt_files}")
    for txt_file in txt_files:
        file_path = os.path.join(PRE_TRAINED_DATA_DIR, txt_file)
        try:
            loader = TextLoader(file_path, encoding="utf-8")
            documents = loader.load()
            all_documents.extend(documents)
            print(f"Loaded: {txt_file} ({len(documents)} pages)")
        except Exception as e:
            print(f"Error loading {txt_file}: {e}")
    return all_documents

def create_vector_store(documents, store_name: str = "default"):
    """Create FAISS vector store from documents"""
    if not documents:
        return None
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
    )
    vector_store = FAISS.from_documents(chunks, embeddings)    # Create vector store
    # Save to disk
    store_path = os.path.join(VECTOR_STORE_DIR, store_name)
    vector_store.save_local(store_path)
    print(f"Created vector store with {len(chunks)} chunks")
    return vector_store

def load_vector_store(store_name: str = "default"):
    """Load existing vector store"""
    store_path = os.path.join(VECTOR_STORE_DIR, store_name)
    if not os.path.exists(store_path):
        return None
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
        )
        vector_store = FAISS.load_local(
            store_path, embeddings, allow_dangerous_deserialization=True
        )
        print(f"Loaded existing vector store from '{store_path}'")
        return vector_store
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None

def query_documents(query: str, vector_store, k: int = 3) -> List[str]:
    if not vector_store:
        return []
    results = vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in results]
    
# =========================
# CHAT RESPONSE GENERATOR
# =========================
class RAGChatbot:
    def __init__(self):
        self.vector_store = None
        self.initialize_vector_store()

    def initialize_vector_store(self):
        """Initialize vector store with pre-trained data or load existing"""
        self.vector_store = load_vector_store()    # Try to load existing vector store first
        # If no vector store exists, create one from pre-trained data
        if not self.vector_store:
            print("No existing vector store found. Creating from pre-trained data...")
            pre_trained_docs = load_all_training_documents()
            if pre_trained_docs:
                self.vector_store = create_vector_store(pre_trained_docs, "default")
                print(
                    f"Vector store created with {len(pre_trained_docs)} pre-trained documents"
                )
            else:
                print(
                    "No pre-trained documents found. Vector store will be empty until documents are uploaded."
                )
                print(
                    f" Place your TXT files in the '{PRE_TRAINED_DATA_DIR}' folder for pre-training"
                )

    def update_vector_store(self, documents):
        """Update vector store with new documents"""
        if not self.vector_store:
            self.vector_store = create_vector_store(documents, "default")    # If no vector store exists, create new one 
        else:
            # If vector store exists, add new documents to it
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            chunks = text_splitter.split_documents(documents)
            self.vector_store.add_documents(chunks)    # Add to existing vector store
            store_path = os.path.join(VECTOR_STORE_DIR, "default")     # Save updated vector store
            self.vector_store.save_local(store_path)
            print(f"Added {len(chunks)} new chunks to vector store")

    def get_response(self, question: str) -> str:
        """Get response combining greetings and document knowledge"""
        question_lower = question.lower().strip()
        # ============================================
        # 1. QUICK DETECTION FOR GENERAL CHAT
        # Bypass vector search for simple greetings/small talk to improve response speed.
        # ============================================
        general_patterns = [
            r"\bhi+\b", r"\bhello\b", r"\bhey\b", r"\bgreetings\b", 
            r"\bgood (morning|afternoon|evening|night)\b",
            r"\bhow are you\b", r"\bwhat's up\b", r"\bhow is it going\b",
            r"\bthank(s| you)?\b", r"\bbye\b", r"\bgoodbye\b", r"\bsee you\b",
            r"\bwho are you\b", r"\bwhat are you\b", r"\byour name\b"
        ]
        # If matching small talk and the query is short, respond using the LLM without document context.
        if any(re.search(p, question_lower) for p in general_patterns) and len(question_lower.split()) < 10:
            prompt = f"""You are a friendly and professional assistant for . 
            Respond naturally and very briefly to this general message: "{question}"
            Answer: "" "
        return llm.invoke(prompt)
        # ============================================
        # 2. DOCUMENT-BASED Q&A (RAG)
        # ============================================
        if self.vector_store:
            try:
                relevant_chunks = query_documents(question, self.vector_store, k=3)
                if relevant_chunks:
                    context = "\n\n".join(relevant_chunks)
                    prompt = f""" You are a helpful assistant for .
                                    Use the context below to answer the question.
                                    
                                    Context:
                                    {context}
                                    
                                    Question: {question}
                                    
                                    Instructions:
                                     - If the answer exists in the context, answer clearly and accurately using ONLY information from the context.
                                     - If the question is general conversation (greetings, good night, jokes), answer it naturally.
                                     - If the question is specific but unrelated to the context, say: "I don't have information about that in my documents."
                                     - NEVER mention any filenames, document names, or source references in your answer.
                                    
                                    Answer:"""
                    return llm.invoke(prompt)
                else:
                    return "I don't have information about that in my documents."
            except Exception as e:
                print(f"Document query error: {e}")
                return "Sorry, I encountered an error while searching the documents."
        return "I don't have information about that in my documents."

# =========================
#  TEXT-TO-SPEECH
# =========================
def text_to_speech(text: str) -> str:
    """Convert text to speech audio"""
    if not TTS_AVAILABLE:
        return ""
    try:
        tts = gTTS(text=text, lang="en", slow=False)
        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
        audio_bytes = buffer.getvalue()
        return base64.b64encode(audio_bytes).decode("utf-8")
    except Exception as e:
        print(f"TTS error: {e}")
        return ""
        
# =========================
#  FASTAPI APP
# =========================
app = FastAPI(title="RAG Voice Chatbot")
chatbot = RAGChatbot()
# Serve static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# =========================
#  API ENDPOINTS
# =========================
@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process document"""
    if not DOCUMENT_AVAILABLE:
        raise HTTPException(
            status_code=400,
            detail="Document processing not available. Install required packages.",
        )
    try:
        file_path = save_uploaded_file(file)     # Save file
        documents = load_documents(file_path)    # Load document
        if not documents:
            raise HTTPException(status_code=400, detail="Failed to load document")
        chatbot.update_vector_store(documents)    # Update vector store
        # Get document info
        num_pages = len(documents) if hasattr(documents, "__len__") else 1
        content_preview = (
            documents[0].page_content[:200] + "..." if documents[0].page_content else ""
        )
        return {
            "success": True,
            "filename": file.filename,
            "file_path": file_path,
            "num_pages": num_pages,
            "preview": content_preview,
            "message": f"Document '{file.filename}' uploaded and learned successfully!",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

@app.post("/reinitialize")
async def reinitialize():
    """Reinitialize chatbot with current training data"""
    chatbot.initialize_vector_store()
    txt_files = []    # Count training files
    if os.path.exists(PRE_TRAINED_DATA_DIR):
        txt_files = [f for f in os.listdir(PRE_TRAINED_DATA_DIR) if f.endswith(".txt")]
    return {
        "success": True,
        "message": "Chatbot reinitialized with training data",
        "training_files": txt_files,
        "has_vector_store": chatbot.vector_store is not None,
    }

@app.post("/chat")
async def chat(question: str = Form(...)):
    """Main chat endpoint with RAG"""
    answer = chatbot.get_response(question)    # Get response from chatbot
    # Generate speech
    audio_base64 = text_to_speech(answer)
    return {
        "question": question,
        "answer": answer,
        "audio_base64": audio_base64,
        "success": True,
    }

@app.get("/document-status")
async def document_status():
    """Check document status including training data"""
    has_documents = chatbot.vector_store is not None
    doc_files = []    # Count files in documents directory
    if os.path.exists(DOCUMENTS_DIR):
        doc_files = [
            f for f in os.listdir(DOCUMENTS_DIR) if f.endswith((".pdf", ".txt"))
        ]
    # Count training files
    training_files = []
    if os.path.exists(PRE_TRAINED_DATA_DIR):
        training_files = [
            f for f in os.listdir(PRE_TRAINED_DATA_DIR) if f.endswith(".txt")
        ]
    return {
        "has_documents": has_documents,
        "document_count": len(doc_files),
        "training_file_count": len(training_files),
        "documents": doc_files[:10],    # Return first 10 files
        "training_files": training_files[:10],
    }

# =========================
#  WEB INTERFACE
# =========================
@app.get("/")
async def index():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# =========================
#  START SERVER
# =========================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
