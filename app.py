"""
RAG VOICE CHATBOT - Document Based with Greetings
-------------------------------------------------------------
Combines: General greetings + Document Q&A + Voice interface
Now with pre-trained document folder support
"""

import os
import io
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

    llm = OllamaLLM(model="qwen3.5:0.8b", temperature=0)
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
DOCUMENTS_DIR = "voice_chatbot_uploaded_documents"
VECTOR_STORE_DIR = "voice_chatbot_vector_store2"
PRE_TRAINED_DATA_DIR = (
    "voice_chatbot_pre_trained_data_bihar"  # Folder containing your existing TXT files
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

    # Create vector store
    vector_store = FAISS.from_documents(chunks, embeddings)

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
        vector_store = FAISS.load_local(store_path, embeddings)
        print(f"Loaded existing vector store from '{store_path}'")
        return vector_store
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None


RELEVANCE_THRESHOLD = 0.45  # tune between 0.35–0.65


def query_documents(query: str, vector_store, k: int = 3) -> List[str]:
    if not vector_store:
        return []
    results = vector_store.similarity_search_with_score(query, k=k)
    # Lower score = more similar in FAISS (L2 distance)
    filtered = [
        doc.page_content for doc, score in results if score <= RELEVANCE_THRESHOLD
    ]
    return filtered


# =========================
# CHAT RESPONSE GENERATOR
# =========================
class RAGChatbot:
    def __init__(self):
        self.vector_store = None
        self.initialize_vector_store()

    def initialize_vector_store(self):
        """Initialize vector store with pre-trained data or load existing"""
        # Try to load existing vector store first
        self.vector_store = load_vector_store()

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
            # If no vector store exists, create new one
            self.vector_store = create_vector_store(documents, "default")
        else:
            # If vector store exists, add new documents to it
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )

            chunks = text_splitter.split_documents(documents)

            # Add to existing vector store
            self.vector_store.add_documents(chunks)

            # Save updated vector store
            store_path = os.path.join(VECTOR_STORE_DIR, "default")
            self.vector_store.save_local(store_path)

            print(f"Added {len(chunks)} new chunks to vector store")

    def get_response(self, question: str) -> str:
        """Get response combining greetings and document knowledge"""

        question_lower = question.lower().strip()

        # ============================================
        # 1. GENERAL GREETINGS & CONVERSATION
        # ============================================

        # GREETINGS
        greetings = {
            "hi": [
                "Hello! 👋How can I help you today?",
                "Hi there! Ready to assist you.",
                "Hey! What can I do for you?",
            ],
            "hello": [
                "Hello! 👋 How can I help you today?",
                "Hi there! Ready to assist you.",
                "Hey! What can I do for you?",
            ],
            "hey": ["Hey! 👋 What's up?", "Hello there! How are you?"],
            "greetings": ["Greetings! How may I assist you today?"],
            "good morning": ["Good morning! 🌞 A great day to learn something new!"],
            "good afternoon": ["Good afternoon! ☀️ How can I help you?"],
            "good evening": ["Good evening! 🌙 Ready to answer your questions."],
            "how are you": [
                "I'm doing great, thanks for asking! How about you?",
                "I'm functioning perfectly! How can I assist you today?",
            ],
            "what's up": [
                "Just here waiting to help you! What's up with you?",
                "All good! What can I do for you today?",
            ],
            "how is it going": [
                "Going well! Ready to help with your questions.",
                "Everything's working perfectly! How about you?",
            ],
        }

        for key, responses in greetings.items():
            if key in question_lower:
                import random

                return random.choice(responses)

        # HELP & INSTRUCTIONS
        if any(
            word in question_lower
            for word in ["help", "how to use", "instructions", "what can you do"]
        ):
            return """**I'm a RAG Voice Chatbot!** 

                **Voice Features:**
                • Speak to me (click microphone)
                • I'll respond with voice
                • Real-time speech recognition

                 **Document Intelligence:**
                • I'm pre-trained on your TXT files in the 'data' folder
                • Ask questions about document content
                • Get accurate answers based on training data
                • Supports PDF and TXT file uploads

                **General Chat:**
                • Casual conversation
                • Greetings and small talk
                • General knowledge

                **My Training Data:**
                • Already trained on TXT files in 'data' folder
                • Can learn from new uploads
                • Context-aware responses

        Try asking about your documents or just say hello!"""

        # THANKS
        if any(word in question_lower for word in ["thank", "thanks", "thank you"]):
            responses = [
                "You're welcome! 😊 Happy to help!",
                "Glad I could assist! 👍",
                "Anytime! Let me know if you need more help.",
                "You're welcome! Don't hesitate to ask more questions.",
            ]
            import random

            return random.choice(responses)

        # GOODBYE
        if any(
            word in question_lower for word in ["bye", "goodbye", "see you", "farewell"]
        ):
            responses = [
                "Goodbye! 👋 Come back anytime!",
                "See you later! Have a great day!",
                "Bye! Feel free to return with more questions.",
                "Farewell! It was nice chatting with you.",
            ]
            import random

            return random.choice(responses)

        # TIME & DATE
        if question_lower in [
            "time",
            "time now",
            "what time is now",
            "what time is it",
            "current time",
            "whats the time",
            "tell me the time",
        ]:
            current_time = datetime.now().strftime("%I:%M %p")
            return f"The current time is {current_time}."

        if "date" in question_lower or "today" in question_lower:
            current_date = datetime.now().strftime("%B %d, %Y")
            return f"Today is {current_date}."

        # WHO ARE YOU / INTRO
        if any(
            phrase in question_lower
            for phrase in ["who are you", "what are you", "your name"]
        ):
            return "I'm a RAG Voice Assistant! I'm pre-trained on your TXT documents. I can chat with you and answer questions from both my training data and newly uploaded documents!"

        # ============================================
        # 2. DOCUMENT-BASED Q&A (RAG)
        # ============================================

        if self.vector_store:
            try:
                relevant_chunks = query_documents(
                    question, self.vector_store, k=3
                )  # Get relevant document chunks

                if relevant_chunks:
                    context = "\n\n".join(relevant_chunks)  # Fix 6: use all chunks
                    prompt = f"""You are a document-based assistant. You ONLY answer from the context provided below.
                         Context:
                        {context}
                 
                        Question: {question}
                        
                        Rules:
                        - If the answer is clearly found in the context, answer it accurately.
                        - If the question is unrelated to the context (like food, general knowledge, etc.), respond ONLY with: "I don't have information about that in my documents."
                        - NEVER use outside knowledge. NEVER make up answers.
                        - ONLY use words and facts from the context above.
                        
                        Answer:"""
                    return llm.invoke(prompt)  # Fix 1: real LLM answer
                else:
                    return "I don't have information about that in my documents. Please ask something related to the uploaded content."

            except Exception as e:
                print(f"Document query error: {e}")
                return "Sorry, I encountered an error while searching the documents."

        # ============================================
        # 3. DEFAULT/GENERAL RESPONSES
        # ============================================

        # Check if asking about training data
        if any(
            word in question_lower for word in ["trained", "training", "learn", "data"]
        ):
            # Count training files
            if os.path.exists(PRE_TRAINED_DATA_DIR):
                txt_files = [
                    f for f in os.listdir(PRE_TRAINED_DATA_DIR) if f.endswith(".txt")
                ]
                return f"I've been trained on {len(txt_files)} TXT files from the 'data' folder. I can answer questions based on their content and learn from new documents you upload!"
            else:
                return (
                    "I'm ready to learn! You can upload documents for me to train on."
                )

        # Upload/document related
        if any(
            word in question_lower
            for word in ["upload", "document", "file", "pdf", "txt"]
        ):
            return "You can upload PDF or TXT documents using the upload button above. Once uploaded, I'll incorporate them into my knowledge and answer questions based on their content!"

        # Weather
        if "weather" in question_lower:
            return "I don't have real-time weather data, but I can help with questions about my training documents or have a friendly chat!"

        # Jokes
        if any(word in question_lower for word in ["joke", "funny"]):
            jokes = [
                "Why don't scientists trust atoms? Because they make up everything!",
                "Why did the scarecrow win an award? He was outstanding in his field!",
                "What do you call fake spaghetti? An impasta!",
                "Why don't eggs tell jokes? They'd crack each other up!",
            ]
            import random

            return random.choice(jokes)

        # Default responses
        default_responses = [
            f"I heard: '{question}'. That's interesting! I can help you with information from my training documents or general conversation.",
            f"Regarding '{question}', I can provide answers based on my training data or we can chat generally.",
            f"Thanks for asking about '{question}'. I've been trained on TXT documents and can learn from new ones you upload!",
            f"Good question! I can answer based on my training data from the 'data' folder. Upload more documents to expand my knowledge!",
            f"I understand you're asking about '{question}'. As a RAG assistant, I combine document knowledge with conversation!",
        ]

        import random

        return random.choice(default_responses)


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
        # Save file
        file_path = save_uploaded_file(file)

        # Load document
        documents = load_documents(file_path)

        if not documents:
            raise HTTPException(status_code=400, detail="Failed to load document")

        # Update vector store
        chatbot.update_vector_store(documents)

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

    # Count training files
    txt_files = []
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

    # Get response from chatbot
    answer = chatbot.get_response(question)

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

    # Count files in documents directory
    doc_files = []
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
        "documents": doc_files[:10],  # Return first 10 files
        "training_files": training_files[:10],
    }


# =========================
#  WEB INTERFACE
# =========================
@app.get("/")
async def home():
    """Main web interface"""

    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Voice Chatbot - Pre-trained Document Intelligence</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            
            body {
                background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
                min-height: 100vh;
                padding: 20px;
                display: flex;
                justify-content: center;
                align-items: center;
            }
            
            .container {
                width: 100%;
                max-width: 900px;
                background: white;
                border-radius: 20px;
                padding: 30px;
                box-shadow: 0 20px 50px rgba(0,0,0,0.2);
                display: flex;
                flex-direction: column;
                gap: 25px;
            }
            
            .header {
                text-align: center;
                padding-bottom: 20px;
                border-bottom: 2px solid #f0f0f0;
            }
            
            .header h1 {
                color: #333;
                margin-bottom: 10px;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 15px;
                font-size: 28px;
            }
            
            .header .tagline {
                color: #666;
                font-size: 16px;
                line-height: 1.5;
            }
            
            .features {
                display: flex;
                flex-wrap: wrap;
                gap: 15px;
                justify-content: center;
                margin: 20px 0;
            }
            
            .feature {
                background: #f8f9fa;
                border-radius: 10px;
                padding: 15px;
                text-align: center;
                flex: 1;
                min-width: 180px;
                border: 1px solid #e9ecef;
            }
            
            .feature i {
                font-size: 24px;
                color: #2575fc;
                margin-bottom: 10px;
            }
            
            .feature h3 {
                color: #333;
                font-size: 16px;
                margin-bottom: 5px;
            }
            
            .feature p {
                color: #666;
                font-size: 13px;
            }
            
            .main-content {
                display: flex;
                flex-direction: column;
                gap: 25px;
            }
            
            .upload-section {
                background: #f8f9fa;
                border-radius: 15px;
                padding: 25px;
                border: 2px dashed #ddd;
                transition: all 0.3s;
            }
            
            .upload-section:hover {
                border-color: #2575fc;
                background: #f0f7ff;
            }
            
            .upload-section h2 {
                color: #333;
                margin-bottom: 15px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .upload-area {
                display: flex;
                gap: 15px;
                align-items: center;
                flex-wrap: wrap;
            }
            
            .file-input {
                flex: 1;
                padding: 15px;
                border: 2px solid #ddd;
                border-radius: 10px;
                font-size: 16px;
            }
            
            .upload-btn {
                padding: 15px 30px;
                background: #28a745;
                color: white;
                border: none;
                border-radius: 10px;
                cursor: pointer;
                font-size: 16px;
                display: flex;
                align-items: center;
                gap: 10px;
                transition: 0.3s;
            }
            
            .upload-btn:hover {
                background: #218838;
                transform: translateY(-2px);
            }
            
            .reinit-btn {
                padding: 15px 25px;
                background: #6f42c1;
                color: white;
                border: none;
                border-radius: 10px;
                cursor: pointer;
                font-size: 16px;
                display: flex;
                align-items: center;
                gap: 10px;
                transition: 0.3s;
            }
            
            .reinit-btn:hover {
                background: #5a2d9c;
                transform: translateY(-2px);
            }
            
            .upload-status {
                margin-top: 15px;
                padding: 12px;
                border-radius: 8px;
                display: none;
            }
            
            .upload-status.success {
                background: #d4edda;
                color: #155724;
                display: block;
            }
            
            .upload-status.error {
                background: #f8d7da;
                color: #721c24;
                display: block;
            }
            
            .chat-interface {
                display: flex;
                flex-direction: column;
                gap: 20px;
            }
            
            .chat-box {
                height: 350px;
                overflow-y: auto;
                border: 1px solid #ddd;
                border-radius: 15px;
                padding: 20px;
                background: #f9f9f9;
            }
            
            .message {
                margin: 15px 0;
                padding: 15px 20px;
                border-radius: 15px;
                max-width: 85%;
                word-wrap: break-word;
                line-height: 1.5;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            
            .user {
                background: linear-gradient(135deg, #2575fc 0%, #6a11cb 100%);
                color: white;
                margin-left: auto;
                text-align: right;
            }
            
            .bot {
                background: #f0f0f0;
                color: #333;
                margin-right: auto;
                border-left: 4px solid #2575fc;
            }
            
            .bot.trained {
                border-left: 4px solid #28a745;
            }
            
            .bot.uploaded {
                border-left: 4px solid #6f42c1;
            }
            
            .controls {
                display: flex;
                flex-direction: column;
                gap: 15px;
            }
            
            .voice-controls {
                display: flex;
                gap: 15px;
                align-items: center;
            }
            
            .voice-btn {
                padding: 18px 25px;
                background: #dc3545;
                color: white;
                border: none;
                border-radius: 12px;
                font-size: 16px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 12px;
                flex: 1;
                transition: all 0.3s;
            }
            
            .voice-btn:hover {
                background: #c82333;
                transform: translateY(-2px);
            }
            
            .voice-btn.recording {
                background: #ffc107;
                color: #000;
                animation: pulse 1.5s infinite;
            }
            
            .text-input-area {
                display: flex;
                gap: 15px;
            }
            
            #textInput {
                flex: 1;
                padding: 18px;
                border: 2px solid #ddd;
                border-radius: 12px;
                font-size: 16px;
                transition: border 0.3s;
            }
            
            #textInput:focus {
                border-color: #2575fc;
                outline: none;
                box-shadow: 0 0 0 3px rgba(37, 117, 252, 0.1);
            }
            
            .send-btn {
                padding: 18px 35px;
                background: #28a745;
                color: white;
                border: none;
                border-radius: 12px;
                cursor: pointer;
                font-size: 16px;
                display: flex;
                align-items: center;
                gap: 10px;
                transition: all 0.3s;
            }
            
            .send-btn:hover {
                background: #218838;
                transform: translateY(-2px);
            }
            
            .status {
                padding: 15px;
                border-radius: 10px;
                text-align: center;
                font-size: 14px;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 10px;
            }
            
            .status-ready {
                background: #d4f7e0;
                color: #2d7a3d;
            }
            
            .status-recording {
                background: #fff3cd;
                color: #856404;
            }
            
            .status-processing {
                background: #d1ecf1;
                color: #0c5460;
            }
            
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.7; }
            }
            
            .samples {
                margin-top: 20px;
                text-align: center;
            }
            
            .sample-buttons {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                justify-content: center;
                margin-top: 10px;
            }
            
            .sample-btn {
                padding: 10px 20px;
                background: #f0f0f0;
                border: 1px solid #ddd;
                border-radius: 20px;
                font-size: 14px;
                cursor: pointer;
                transition: all 0.2s;
            }
            
            .sample-btn:hover {
                background: #e0e0e0;
                transform: translateY(-2px);
            }
            
            .document-status {
                background: #e8f4fc;
                border-radius: 10px;
                padding: 15px;
                margin-top: 20px;
                display: none;
            }
            
            .document-status.show {
                display: block;
            }
            
            .training-info {
                background: #d4f7e0;
                border-radius: 10px;
                padding: 15px;
                margin-top: 20px;
            }
            
            .instructions {
                font-size: 13px;
                color: #888;
                text-align: center;
                margin-top: 20px;
                line-height: 1.6;
            }
            
            .footer {
                text-align: center;
                padding-top: 20px;
                border-top: 1px solid #eee;
                color: #666;
                font-size: 13px;
            }
            
            .message-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 8px;
                font-size: 12px;
                opacity: 0.8;
            }
            
            .document-indicator {
                background: #28a745;
                color: white;
                padding: 3px 8px;
                border-radius: 10px;
                font-size: 11px;
            }
            
            .training-indicator {
                background: #6f42c1;
                color: white;
                padding: 3px 8px;
                border-radius: 10px;
                font-size: 11px;
            }
            
            .upload-indicator {
                background: #fd7e14;
                color: white;
                padding: 3px 8px;
                border-radius: 10px;
                font-size: 11px;
            }
        </style>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>
                    <i class="fas fa-robot"></i>
                    RAG Voice Chatbot
                    <i class="fas fa-graduation-cap"></i>
                </h1>
                <div class="tagline">
                    Pre-trained on your TXT documents + General conversation + Voice interface
                </div>
                
                <div class="features">
                    <div class="feature">
                        <i class="fas fa-microphone"></i>
                        <h3>Voice Interface</h3>
                        <p>Speak naturally</p>
                    </div>
                    <div class="feature">
                        <i class="fas fa-graduation-cap"></i>
                        <h3>Pre-trained</h3>
                        <p>Learned from TXT files</p>
                    </div>
                    <div class="feature">
                        <i class="fas fa-file-pdf"></i>
                        <h3>Document Q&A</h3>
                        <p>Ask about documents</p>
                    </div>
                    <div class="feature">
                        <i class="fas fa-comments"></i>
                        <h3>General Chat</h3>
                        <p>Casual conversation</p>
                    </div>
                </div>
            </div>
            
            <div class="training-info" id="trainingInfo">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong><i class="fas fa-graduation-cap"></i> Training Status:</strong>
                        <span id="trainingStatus">Loading...</span>
                    </div>
                    <button class="reinit-btn" onclick="reinitializeBot()">
                        <i class="fas fa-sync-alt"></i> Reinitialize
                    </button>
                </div>
            </div>
            
            <div class="main-content">
                <!-- Upload Section -->
                <div class="upload-section">
                    <h2><i class="fas fa-cloud-upload-alt"></i> Upload New Documents</h2>
                    <div class="upload-area">
                        <input type="file" id="fileInput" class="file-input" accept=".pdf,.txt" multiple>
                        <button class="upload-btn" onclick="uploadDocuments()">
                            <i class="fas fa-upload"></i> Upload & Learn
                        </button>
                    </div>
                    <div id="uploadStatus" class="upload-status"></div>
                </div>
                
                <!-- Document Status -->
                <div id="documentStatus" class="document-status">
                    <div id="statusContent"></div>
                </div>
                
                <!-- Chat Interface -->
                <div class="chat-interface">
                    <div class="chat-box" id="chatBox">
                        <div class="message bot trained">
                            <div class="message-header">
                                <span><i class="fas fa-robot"></i> Assistant</span>
                                <span class="training-indicator">Pre-trained Assistant</span>
                            </div>
                            Hello! I'm your <strong>pre-trained</strong> RAG Voice Assistant. 
                            <br><br>
                            I've been trained on TXT files from the 'data' folder and can:
                            <br><br>
                            1. <strong>Answer questions</strong> based on my training data
                            <br>
                            2. <strong>Learn from new documents</strong> you upload (PDF/TXT)
                            <br>
                            3. <strong>Have general conversations</strong> (greetings, jokes, etc.)
                            <br>
                            4. <strong>Speak and listen</strong> using voice interface
                            <br><br>
                            Try asking about your documents or just start chatting!
                        </div>
                    </div>
                    
                    <div class="controls">
                        <div class="voice-controls">
                            <button class="voice-btn" id="voiceBtn" onclick="toggleVoiceRecognition()">
                                <i class="fas fa-microphone"></i>
                                <span id="voiceBtnText">Start Speaking</span>
                            </button>
                        </div>
                        
                        <div class="status status-ready" id="status">
                            <i class="fas fa-info-circle"></i>
                            <span id="statusText">Ready! I'm pre-trained on your documents.</span>
                        </div>
                        
                        <div class="text-input-area">
                            <input type="text" id="textInput" 
                                   placeholder="Ask about your documents or just chat..." 
                                   onkeypress="if(event.key === 'Enter') sendMessage()">
                            <button class="send-btn" onclick="sendMessage()">
                                <i class="fas fa-paper-plane"></i> Send
                            </button>
                        </div>
                    </div>
                    
                    <div class="samples">
                        <div style="font-size: 13px; color: #666; margin-bottom: 10px;">
                            <i class="fas fa-lightbulb"></i> Try these examples:
                        </div>
                        <div class="sample-buttons">
                            <button class="sample-btn" onclick="useExample('Hello! How are you?')">
                                <i class="fas fa-hand-wave"></i> Greeting
                            </button>
                            <button class="sample-btn" onclick="useExample('What are you trained on?')">
                                <i class="fas fa-graduation-cap"></i> Training
                            </button>
                            <button class="sample-btn" onclick="useExample('What time is it?')">
                                <i class="fas fa-clock"></i> Time
                            </button>
                            <button class="sample-btn" onclick="useExample('Tell me a joke')">
                                <i class="fas fa-laugh"></i> Joke
                            </button>
                            <button class="sample-btn" onclick="useExample('Ask about document content')">
                                <i class="fas fa-file"></i> Documents
                            </button>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="instructions">
                <i class="fas fa-info-circle"></i> 
                <strong>How it works:</strong> 
                I'm pre-trained on TXT files in the 'data' folder → You can upload more documents → Ask questions based on all content + general chat
            </div>
            
            <div class="footer">
                <p>RAG Voice Chatbot • Pre-trained on your documents • Learns from new uploads • Voice & text interface</p>
            </div>
        </div>
        
        <script>
            // Speech recognition variables
            let recognition = null;
            let isListening = false;
            
            // DOM elements
            const chatBox = document.getElementById('chatBox');
            const textInput = document.getElementById('textInput');
            const voiceBtn = document.getElementById('voiceBtn');
            const voiceBtnText = document.getElementById('voiceBtnText');
            const status = document.getElementById('status');
            const statusText = document.getElementById('statusText');
            const uploadStatus = document.getElementById('uploadStatus');
            const documentStatus = document.getElementById('documentStatus');
            const statusContent = document.getElementById('statusContent');
            const trainingInfo = document.getElementById('trainingInfo');
            const trainingStatus = document.getElementById('trainingStatus');
            
            // Check browser support
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            
            if (SpeechRecognition) {
                recognition = new SpeechRecognition();
                recognition.continuous = false;
                recognition.interimResults = true;
                recognition.lang = 'en-US';
                
                recognition.onstart = function() {
                    console.log("Voice recognition started");
                    isListening = true;
                    voiceBtn.classList.add('recording');
                    voiceBtnText.textContent = "Stop Speaking";
                    updateStatus(" Listening... Speak clearly into microphone", 'recording');
                };
                
                recognition.onresult = function(event) {
                    let finalTranscript = '';
                    let interimTranscript = '';
                    
                    for (let i = event.resultIndex; i < event.results.length; i++) {
                        const transcript = event.results[i][0].transcript;
                        if (event.results[i].isFinal) {
                            finalTranscript += transcript;
                        } else {
                            interimTranscript += transcript;
                        }
                    }
                    
                    // Update input with recognized text
                    if (finalTranscript) {
                        textInput.value = finalTranscript;
                    } else if (interimTranscript) {
                        textInput.value = interimTranscript;
                    }
                };
                
                recognition.onerror = function(event) {
                    console.error("Speech recognition error:", event.error);
                    
                    if (event.error === 'not-allowed') {
                        updateStatus(" Microphone access denied. Please allow permission.", 'error');
                    } else if (event.error === 'no-speech') {
                        updateStatus(" No speech detected. Please speak louder.", 'error');
                    } else {
                        updateStatus(" Speech recognition error: " + event.error, 'error');
                    }
                    
                    stopListening();
                };
                
                recognition.onend = function() {
                    console.log("Voice recognition ended");
                    if (isListening) {
                        stopListening();
                        
                        // Auto-send if we have text
                        if (textInput.value.trim()) {
                            setTimeout(() => {
                                sendMessage();
                            }, 800);
                        }
                    }
                };
            } else {
                voiceBtn.disabled = true;
                voiceBtnText.textContent = "Voice Not Supported";
                voiceBtn.style.background = "#95a5a6";
                updateStatus(" Voice recognition not supported in this browser. Please type.", 'error');
            }
            
            // Add message to chat
            function addMessage(text, isUser = false, messageType = 'general') {
                const div = document.createElement('div');
                div.className = `message ${isUser ? 'user' : 'bot'}`;
                
                if (!isUser) {
                    if (messageType === 'trained') {
                        div.classList.add('trained');
                    } else if (messageType === 'uploaded') {
                        div.classList.add('uploaded');
                    }
                }
                
                let header = '';
                if (!isUser) {
                    let indicator = '';
                    if (messageType === 'trained') {
                        indicator = `<span class="training-indicator"><i class="fas fa-graduation-cap"></i> From Training</span>`;
                    } else if (messageType === 'uploaded') {
                        indicator = `<span class="upload-indicator"><i class="fas fa-file-upload"></i> From Upload</span>`;
                    } else {
                        indicator = `<span class="document-indicator">General Chat</span>`;
                    }
                    
                    header = `<div class="message-header">
                                <span><i class="fas fa-robot"></i> Assistant</span>
                                ${indicator}
                              </div>`;
                } else {
                    header = `<div class="message-header">
                                <span><i class="fas fa-user"></i> You</span>
                              </div>`;
                }
                
                div.innerHTML = header + text;
                chatBox.appendChild(div);
                chatBox.scrollTop = chatBox.scrollHeight;
            }
            
            // Update status
            function updateStatus(message, type = 'ready') {
                statusText.textContent = message;
                status.className = `status status-${type}`;
            }
            
            // Toggle voice recognition
            function toggleVoiceRecognition() {
                if (!recognition) {
                    updateStatus("Voice recognition not supported", 'error');
                    return;
                }
                
                if (!isListening) {
                    startListening();
                } else {
                    stopListening();
                }
            }
            
            // Start listening
            function startListening() {
                try {
                    textInput.value = '';
                    recognition.start();
                } catch (error) {
                    updateStatus("Cannot start speech recognition", 'error');
                }
            }
            
            // Stop listening
            function stopListening() {
                if (recognition && isListening) {
                    try {
                        recognition.stop();
                    } catch (error) {
                        console.log("Error stopping recognition:", error);
                    }
                }
                
                isListening = false;
                voiceBtn.classList.remove('recording');
                voiceBtnText.textContent = "Start Speaking";
            }
            
            // Send message to server
            async function sendMessage() {
                const text = textInput.value.trim();
                if (!text) {
                    updateStatus("Please type or speak something first", 'error');
                    return;
                }
                
                // Add user message to chat
                addMessage(text, true);
                
                // Clear input
                textInput.value = '';
                
                // Update status
                updateStatus("Processing your question...", 'processing');
                
                try {
                    const formData = new FormData();
                    formData.append('question', text);
                    
                    const response = await fetch('/chat', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        // Determine message type
                        let messageType = 'general';
                        if (result.answer.includes('trained') || result.answer.includes('training')) {
                            messageType = 'trained';
                        } else if (result.answer.includes('upload') || result.answer.includes('document')) {
                            messageType = 'uploaded';
                        }
                        
                        // Add bot response
                        addMessage(result.answer, false, messageType);
                        
                        // Play audio if available
                        if (result.audio_base64) {
                            playAudio(result.audio_base64);
                        }
                        
                        updateStatus("Ready for next question", 'ready');
                    } else {
                        addMessage("Sorry, there was an error processing your request.", false);
                        updateStatus("Ready", 'ready');
                    }
                } catch (error) {
                    console.error("Error:", error);
                    addMessage("Network error. Please check your connection.", false);
                    updateStatus("Ready", 'ready');
                }
            }
            
            // Play audio response
            function playAudio(base64Audio) {
                try {
                    if (!base64Audio) return;
                    
                    const audio = new Audio('data:audio/mp3;base64,' + base64Audio);
                    audio.play().catch(e => {
                        console.log("Could not play audio:", e);
                    });
                } catch (error) {
                    console.log("Audio play error:", error);
                }
            }
            
            // Use example
            function useExample(text) {
                textInput.value = text;
                textInput.focus();
                sendMessage();
            }
            
            // Upload documents
            async function uploadDocuments() {
                const fileInput = document.getElementById('fileInput');
                const files = fileInput.files;
                
                if (!files.length) {
                    uploadStatus.textContent = "Please select at least one file (PDF or TXT)";
                    uploadStatus.className = 'upload-status error';
                    return;
                }
                
                updateStatus("Uploading and learning from document...", 'processing');
                uploadStatus.textContent = "Uploading...";
                uploadStatus.className = 'upload-status';
                uploadStatus.style.display = 'block';
                
                try {
                    // For now, upload first file
                    const file = files[0];
                    const singleFormData = new FormData();
                    singleFormData.append('file', file);
                    
                    const response = await fetch('/upload-document', {
                        method: 'POST',
                        body: singleFormData
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        uploadStatus.textContent = `✅ ${result.message}`;
                        uploadStatus.className = 'upload-status success';
                        
                        // Update document status display
                        documentStatus.classList.add('show');
                        statusContent.innerHTML = `
                            <strong><i class="fas fa-check-circle"></i> Document Learned!</strong>
                            <div style="margin-top: 10px;">
                                <strong>File:</strong> ${result.filename}<br>
                                <strong>Pages:</strong> ${result.num_pages}<br>
                                <strong>Preview:</strong> ${result.preview}
                            </div>
                        `;
                        
                        // Add system message to chat
                        const systemMsg = document.createElement('div');
                        systemMsg.className = 'message bot uploaded';
                        systemMsg.innerHTML = `
                            <div class="message-header">
                                <span><i class="fas fa-robot"></i> System</span>
                                <span class="upload-indicator">Document Learned</span>
                            </div>
                             Document <strong>${result.filename}</strong> learned successfully! 
                            You can now ask questions about its content along with my training data.
                        `;
                        chatBox.appendChild(systemMsg);
                        chatBox.scrollTop = chatBox.scrollHeight;
                        
                        updateStatus(" New document learned! Ask questions about it.", 'ready');
                        
                    } else {
                        uploadStatus.textContent = ` Upload failed: ${result.detail || 'Unknown error'}`;
                        uploadStatus.className = 'upload-status error';
                        updateStatus(" Upload failed", 'error');
                    }
                    
                } catch (error) {
                    console.error("Upload error:", error);
                    uploadStatus.textContent = " Upload failed. Please try again.";
                    uploadStatus.className = 'upload-status error';
                    updateStatus(" Upload failed", 'error');
                }
                
                // Clear file input
                fileInput.value = '';
            }
            
            // Reinitialize bot
            async function reinitializeBot() {
                try {
                    updateStatus(" Reinitializing with training data...", 'processing');
                    
                    const response = await fetch('/reinitialize', {
                        method: 'POST'
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        updateStatus(" Reinitialized successfully!", 'ready');
                        
                        // Update training status
                        trainingStatus.innerHTML = `
                            Trained on <strong>${result.training_files.length}</strong> files: 
                            ${result.training_files.join(', ')}
                        `;
                        
                        // Add system message
                        const systemMsg = document.createElement('div');
                        systemMsg.className = 'message bot trained';
                        systemMsg.innerHTML = `
                            <div class="message-header">
                                <span><i class="fas fa-robot"></i> System</span>
                                <span class="training-indicator">Reinitialized</span>
                            </div>
                            Reinitialized with training data! I'm now working with ${result.training_files.length} training files.
                        `;
                        chatBox.appendChild(systemMsg);
                        chatBox.scrollTop = chatBox.scrollHeight;
                    }
                } catch (error) {
                    console.error("Reinitialization error:", error);
                    updateStatus(" Reinitialization failed", 'error');
                }
            }
            
            // Check document status on load
            async function checkDocumentStatus() {
                try {
                    const response = await fetch('/document-status');
                    const result = await response.json();
                    
                    // Update training info
                    trainingStatus.innerHTML = `
                        ${result.training_file_count} training files | ${result.document_count} uploaded files
                    `;
                    
                    if (result.has_documents && (result.document_count > 0 || result.training_file_count > 0)) {
                        documentStatus.classList.add('show');
                        
                        let content = `<strong><i class="fas fa-database"></i> Knowledge Base</strong><div style="margin-top: 10px;">`;
                        
                        if (result.training_file_count > 0) {
                            content += `<strong>Training Files (${result.training_file_count}):</strong> ${result.training_files.join(', ')}<br>`;
                        }
                        
                        if (result.document_count > 0) {
                            content += `<strong>Uploaded Files (${result.document_count}):</strong> ${result.documents.join(', ')}`;
                        }
                        
                        content += `</div>`;
                        statusContent.innerHTML = content;
                    }
                } catch (error) {
                    console.log("Could not check document status:", error);
                    trainingStatus.textContent = "Error checking status";
                }
            }
            
            // Initialize
            document.addEventListener('DOMContentLoaded', () => {
                console.log("Pre-trained RAG Voice Chatbot Ready");
                textInput.focus();
                
                // Check document status
                checkDocumentStatus();
                
                // Auto-focus on input when clicking
                document.addEventListener('click', () => {
                    textInput.focus();
                });
            });
        </script>
    </body>
    </html>
    """

    return HTMLResponse(content=html)


# =========================
#  START SERVER
# =========================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" RAG VOICE CHATBOT - Pre-trained Document Intelligence")
    print("=" * 60)

    print("\n REQUIRED PACKAGES:")
    print("   pip install fastapi uvicorn gtts")
    print("   pip install langchain faiss-cpu pypdf sentence-transformers")

    print(f"\n PRE-TRAINING DATA FOLDER: '{PRE_TRAINED_DATA_DIR}'")
    print("   • Place your TXT files in this folder")
    print("   • Chatbot will automatically train on them")
    print("   • Supports additional PDF/TXT uploads")

    print("\n FEATURES:")
    print("   1.  Pre-trained on your TXT documents")
    print("   2.  Voice interface (speak & listen)")
    print("   3.  Document-based Q&A (RAG)")
    print("   4.  General conversation")
    print("   5.  Learn from new uploads")

    print("\n OPEN IN BROCHATBOT IS READY:")
    print("   http://localhost:8000")

    print("\n HOW TO USE:")
    print(f"   1. Place TXT files in '{PRE_TRAINED_DATA_DIR}' folder")
    print("   2. Start the server - chatbot auto-trains")
    print("   3. Ask questions about document content")
    print("   4. Upload more documents to expand knowledge")
    print("   5. Use voice interface or text chat")

    print("\n SUPPORTED FILES:")
    print("   • TXT files (for pre-training)")
    print("   • PDF files (upload)")
    print("   • Additional TXT files (upload)")

    print("\n EXAMPLE QUESTIONS:")
    print("   • 'Hello! How are you?' (General chat)")
    print("   • 'What are you trained on?' (Training info)")
    print("   • 'Ask about [topic] from documents' (Document Q&A)")
    print("   • 'What time is it?' (General knowledge)")
    print("   • 'Tell me a joke' (Casual conversation)")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8013, reload=True)
