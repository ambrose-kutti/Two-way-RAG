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

// Check browser support
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onstart = function () {
        console.log("Voice recognition started");
        isListening = true;
        voiceBtn.classList.add('recording');
        voiceBtnText.textContent = "Stop Speaking";
        updateStatus(" Listening... Speak clearly into microphone", 'recording');
    };

    recognition.onresult = function (event) {
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

    recognition.onerror = function (event) {
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

    recognition.onend = function () {
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
    saveChatToStorage('user', text);
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
            saveChatToStorage('bot', result.answer, messageType);
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

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log("Pre-trained RAG Voice Chatbot Ready");
    textInput.focus();
    renderHistoryList();
    // Auto-focus on input when clicking
    document.addEventListener('click', () => {
        textInput.focus();
    });
});

// Toggle Sidebar
function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    if (sidebar) {
        sidebar.classList.toggle('collapsed');
    }
}

// --- Chat History Features ---
let chatHistory = JSON.parse(localStorage.getItem('rag_chat_history')) || [];
let currentSessionId = null;

function renderHistoryList() {
    const historyList = document.getElementById('historyList');
    if (!historyList) return;
    historyList.innerHTML = '';
    if (chatHistory.length === 0) {
        historyList.innerHTML = '<div style="padding: 1rem; color: #9CA3AF; font-size: 0.8rem; text-align: center;">No previous chats</div>';
        return;
    }

    chatHistory.forEach(session => {
        const div = document.createElement('div');
        div.className = `history-item ${session.id === currentSessionId ? 'active' : ''}`;
        div.innerHTML = `
            <i class="far fa-comment-alt" style="opacity: 0.7; flex-shrink: 0;"></i>
            <span>${session.title}</span>
            <button class="delete-history-btn" title="Delete Chat" onclick="deleteSession('${session.id}', event)">
                <i class="fa-solid fa-trash-can"></i>
            </button>
        `;
        div.onclick = () => loadSession(session.id);
        historyList.appendChild(div);
    });
}

function saveChatToStorage(role, text, type = 'general') {
    if (!currentSessionId) {
        currentSessionId = Date.now().toString();
        const newSession = {
            id: currentSessionId,
            title: text.length > 30 ? text.substring(0, 30) + '...' : text,
            timestamp: Date.now(),
            messages: []
        };
        chatHistory.unshift(newSession);
    }

    // Find session and update
    const session = chatHistory.find(s => s.id === currentSessionId);
    if (session) {
        session.messages.push({ role, text, type, timestamp: Date.now() });
        session.timestamp = Date.now();
        // Move to top of list
        chatHistory = chatHistory.filter(s => s.id !== currentSessionId);
        chatHistory.unshift(session);
        localStorage.setItem('rag_chat_history', JSON.stringify(chatHistory));
        renderHistoryList();
    }
}

function loadSession(sessionId) {
    const session = chatHistory.find(s => s.id === sessionId);
    if (!session) return;
    currentSessionId = sessionId;
    const chatBox = document.getElementById('chatBox');
    chatBox.innerHTML = '';
    session.messages.forEach(msg => {
        addMessage(msg.text, msg.role === 'user', msg.type);
    });
    renderHistoryList(); // Ensure "active" class updates
}

function startNewChat() {
    currentSessionId = null;
    renderHistoryList();
    const chatBox = document.getElementById('chatBox');
    chatBox.innerHTML = '';
    const div = document.createElement('div');
    div.className = 'message bot';
    div.innerHTML = `
        <div class="message-header">
            <i class="fas fa-robot"></i> Assistant
        </div>
        Hello! I'm your <strong>pre-trained</strong> RAG Voice Assistant.<br><br>
        I can answer questions based on your documents, learn from new uploads, or just have a chat. Use the microphone to speak!
    `;
    chatBox.appendChild(div);
}

function deleteSession(sessionId, event) {
    if (event) event.stopPropagation(); // Prevent loading the session
    if (!confirm('Are you sure you want to delete this chat?')) return;
    chatHistory = chatHistory.filter(s => s.id !== sessionId);
    localStorage.setItem('rag_chat_history', JSON.stringify(chatHistory));
    if (currentSessionId === sessionId) {
        startNewChat();
    } else {
        renderHistoryList();
    }
}