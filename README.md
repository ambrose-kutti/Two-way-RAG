# 🗨️Two-way-RAG 🗣️

An interactive Retrieval Augmented Generation (RAG) system with a voice-enabled web interface, powered by FastAPI, LangChain, and Ollama. This application allows you to build and chat with a knowledge base created from your own documents.

## 📖 Overview

This project provides a complete, runnable RAG chatbot application. It features a modern web interface where users can interact with a Large Language Model (LLM) that has been augmented with information from a custom knowledge base. You can load documents into the system through a pre-trained data folder or by uploading them directly via the web UI.

The system uses a vector store to perform semantic searches on your documents, retrieves the most relevant context, and then uses an LLM to generate a coherent, factually grounded answer. It also includes speech-to-text for voice input and text-to-speech for audio responses, creating a dynamic conversational experience.

## ✨ Features

-   **Interactive Web Interface**:  A clean and responsive chat interface built with HTML, CSS, and vanilla JavaScript.
-   **Document Ingestion**:  Upload PDF and TXT files directly through the UI or pre-load a knowledge base by placing TXT files in a dedicated folder.
-   **Semantic Retrieval**:  Utilizes FAISS and Hugging Face sentence transformers for efficient, context-aware document retrieval.
-   **LLM Integration**:  Seamlessly integrates with local LLMs via Ollama to generate context-aware responses.
-   **Voice Interaction**: Supports voice-to-text input using the browser's SpeechRecognition API and provides text-to-speech audio output for responses.
-   **Chat History**:  Automatically saves and loads chat sessions using browser local storage.
-   **Easy Initialization**: A "Reinitialize" feature allows you to rebuild the knowledge base from your pre-trained documents on the fly.

## 🛠️ Tech Stack

**Backend:**
-   Python
-   FastAPI
-   LangChain
-   Ollama (for LLM serving)
-   gTTS (for Text-to-Speech)

**Vector Storage & Embeddings:**
-   FAISS (Facebook AI Similarity Search)
-   Hugging Face Sentence Transformers

**Frontend:**
-   HTML5
-   CSS3
-   JavaScript

## 🚀 Getting Started

Follow these steps to set up and run the Two-way-RAG application on your local machine.

### Prerequisites

-   **Python 3.9+**
-   **Ollama**: Ensure you have [Ollama](https://ollama.com/) installed and running. You also need to have pulled a model. This project is configured to use `llama3.2:latest`.
    ```bash
    ollama pull llama3.2:latest
    ```

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ambrose-kutti/Two-way-RAG.git
    cd Two-way-RAG
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **(Optional) Pre-train with your own data:**
    Place any `.txt` files you want to include in the initial knowledge base into the `pre_trained_data` directory. If this directory doesn't exist, it will be created automatically.

5.  **Run the application:**
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8013 --reload
    ```

6.  **Access the application:**
    Open your web browser and navigate to `http://localhost:8013`.

## ⚙️ How It Works

The application operates through several key components:

1.  **Document Loading**: On startup, the application checks for a pre-existing vector store. If none is found, it loads all `.txt` files from the `pre_trained_data` directory to build an initial knowledge base. You can add more documents (PDF or TXT) anytime using the "Upload & Learn" feature in the UI.

2.  **Vectorization**: Documents are split into smaller chunks. Each chunk is then converted into a numerical vector (embedding) using a Hugging Face model (`all-MiniLM-L6-v2`). These vectors are stored in a FAISS vector store on disk.

3.  **RAG Pipeline**:
    -   When you send a message, the system first checks if it's a general greeting or small talk. If so, it responds directly using the LLM for a faster, more natural conversation.
    -   For specific queries, your question is converted into an embedding.
    -   This embedding is used to search the FAISS vector store for the most semantically similar document chunks (the "context").
    -   A detailed prompt is constructed containing your original question and the retrieved context.
    -   This prompt is sent to the Ollama-served LLM, which generates a response based on the provided information.

4.  **Interaction**: The frontend captures your input (text or speech), sends it to the FastAPI backend, and then displays the text response while playing the generated audio.

## 📁 Project Structure

```
ambrose-kutti-two-way-rag/
├── main.py                 # FastAPI application, RAG logic, and API endpoints
├── requirements.txt        # Python package dependencies
├── static/
│   ├── css/style.css       # Styles for the web interface
│   └── js/script.js        # Frontend logic, speech recognition, and API calls
└── templates/
    └── index.html          # The main HTML file for the user interface
```

## 📚 API Endpoints

The application exposes the following API endpoints:

-   `GET /`: Serves the main web interface.
-   `POST /chat`: Receives a user's question, processes it through the RAG pipeline, and returns a JSON response with the answer and a base64-encoded audio string.
-   `POST /upload-document`: Handles file uploads, processes the document, and adds it to the vector store.
-   `POST /reinitialize`: Clears the current session and rebuilds the vector store using documents from the `pre_trained_data` folder.
-   `GET /document-status`: Provides the status of the knowledge base, including the number of trained and uploaded files.

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements or want to fix a bug, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourFeature`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some feature'`).
5.  Push to the branch (`git push origin feature/YourFeature`).
6.  Open a Pull Request.
