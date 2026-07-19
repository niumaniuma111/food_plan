// Chat application JavaScript

const API_BASE = '/api';
let sessionId = generateSessionId();
let isStreaming = false;

// DOM elements
const chatArea = document.getElementById('chatArea');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const fileInput = document.getElementById('fileInput');

// Initialize marked
marked.setOptions({
    breaks: true,
    gfm: true,
});

// Send message on button click
sendBtn.addEventListener('click', sendMessage);

// Send message on Enter (Shift+Enter for new line)
messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Auto-resize textarea
messageInput.addEventListener('input', () => {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + 'px';
});

// File upload handler
fileInput.addEventListener('change', handleFileUpload);

// Send message
async function sendMessage() {
    const query = messageInput.value.trim();
    if (!query || isStreaming) return;
    
    // Add user message to chat
    addMessage(query, 'user');
    messageInput.value = '';
    messageInput.style.height = 'auto';
    
    // Disable input while streaming
    isStreaming = true;
    sendBtn.disabled = true;
    
    // Create AI message placeholder
    const aiMessageEl = addMessage('', 'ai');
    const contentEl = aiMessageEl.querySelector('.message-content');
    
    try {
        // Start streaming
        const response = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                session_id: sessionId,
            }),
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        // Read stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullResponse = '';
        let sources = null;
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            const text = decoder.decode(value);
            const lines = text.split('\n');
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        
                        if (data.type === 'token') {
                            fullResponse += data.content;
                            contentEl.innerHTML = marked.parse(fullResponse);
                            
                            // Add sources if present
                            if (data.sources && data.sources.length > 0) {
                                sources = data.sources;
                            }
                            
                            // Scroll to bottom
                            chatArea.scrollTop = chatArea.scrollHeight;
                        } else if (data.type === 'done') {
                            // Add sources and feedback buttons
                            if (sources) {
                                addSources(contentEl, sources);
                            }
                            addFeedbackButtons(contentEl, query, fullResponse);
                        } else if (data.type === 'error') {
                            contentEl.innerHTML = `<p style="color: red;">Error: ${data.message}</p>`;
                        }
                    } catch (e) {
                        // Ignore parse errors for incomplete JSON
                    }
                }
            }
        }
        
    } catch (error) {
        contentEl.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
    } finally {
        isStreaming = false;
        sendBtn.disabled = false;
        messageInput.focus();
    }
}

// Add message to chat
function addMessage(content, type) {
    const messageEl = document.createElement('div');
    messageEl.className = `message ${type}-message`;
    
    const contentEl = document.createElement('div');
    contentEl.className = 'message-content';
    contentEl.innerHTML = type === 'user' ? escapeHtml(content) : marked.parse(content);
    
    messageEl.appendChild(contentEl);
    chatArea.appendChild(messageEl);
    chatArea.scrollTop = chatArea.scrollHeight;
    
    return messageEl;
}

// Add sources to message
function addSources(contentEl, sources) {
    const sourcesDiv = document.createElement('div');
    sourcesDiv.className = 'sources';
    sourcesDiv.innerHTML = `
        <div class="sources-title">📚 参考来源:</div>
        ${sources.map(s => `
            <div class="source-item">
                <span class="source-icon">📄</span>
                <span>${escapeHtml(s.filename)}</span>
            </div>
        `).join('')}
    `;
    contentEl.appendChild(sourcesDiv);
}

// Add feedback buttons
function addFeedbackButtons(contentEl, question, answer) {
    const feedbackDiv = document.createElement('div');
    feedbackDiv.className = 'feedback-buttons';
    feedbackDiv.innerHTML = `
        <button class="feedback-btn thumbs-up" onclick="submitFeedback(this, 'positive')">
            👍 有帮助
        </button>
        <button class="feedback-btn thumbs-down" onclick="submitFeedback(this, 'negative')">
            👎 没帮助
        </button>
    `;
    
    // Store question and answer for feedback
    feedbackDiv.dataset.question = question;
    feedbackDiv.dataset.answer = answer;
    
    contentEl.appendChild(feedbackDiv);
}

// Submit feedback
async function submitFeedback(btn, rating) {
    const feedbackDiv = btn.closest('.feedback-buttons');
    const question = feedbackDiv.dataset.question;
    const answer = feedbackDiv.dataset.answer;
    
    // Disable both buttons
    const buttons = feedbackDiv.querySelectorAll('.feedback-btn');
    buttons.forEach(b => b.disabled = true);
    
    // Highlight selected button
    btn.classList.add('submitted');
    
    try {
        const response = await fetch(`${API_BASE}/feedback`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: question,
                answer: answer,
                session_id: sessionId,
                rating: rating,
            }),
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            btn.textContent = rating === 'positive' ? '✓ 已提交审核' : '✓ 已反馈';
        } else if (result.status === 'duplicate') {
            btn.textContent = '已存在相似反馈';
        }
    } catch (error) {
        btn.textContent = '提交失败';
        buttons.forEach(b => b.disabled = false);
    }
}

// Handle file upload
async function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch(`${API_BASE}/documents/upload`, {
            method: 'POST',
            body: formData,
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            addMessage(`📄 文件 "${file.name}" 上传成功！已创建 ${result.chunks_created} 个知识片段。`, 'ai');
        } else {
            addMessage(`❌ 上传失败: ${result.detail || '未知错误'}`, 'ai');
        }
    } catch (error) {
        addMessage(`❌ 上传失败: ${error.message}`, 'ai');
    }
    
    // Reset file input
    fileInput.value = '';
}

// Generate session ID
function generateSessionId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

// Escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Focus input on load
window.addEventListener('load', () => {
    messageInput.focus();
});
