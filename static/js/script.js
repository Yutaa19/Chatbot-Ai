function createMessageElement(content, isUser = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'system'}`;

    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    messageContent.textContent = content;

    messageDiv.appendChild(messageContent);
    return messageDiv;
}

function scrollToBottom() {
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function createTypingIndicator() {
    const indicator = document.createElement('div');
    indicator.className = 'typing-indicator';
    indicator.innerHTML = `
        Mengetik<span></span><span></span><span></span>
    `;
    return indicator;
}

async function sendMessage(event) {
    event.preventDefault();

    const userInput = document.getElementById('userInput');
    const chatMessages = document.getElementById('chatMessages');
    const query = userInput.value.trim();

    if (!query) return;

    // Add user message
    chatMessages.appendChild(createMessageElement(query, true));
    userInput.value = '';
    scrollToBottom();

    // Add typing indicator
    const typingIndicator = createTypingIndicator();
    chatMessages.appendChild(typingIndicator);
    scrollToBottom();

    try {
        const response = await fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query: query })
        });

        const data = await response.json();

        // Remove typing indicator
        typingIndicator.remove();

        if (data.error) {
            chatMessages.appendChild(createMessageElement(
                'Maaf, terjadi kesalahan. Silakan coba lagi.'
            ));
        } else {
            chatMessages.appendChild(createMessageElement(data.answer));
        }

    } catch (error) {
        // Remove typing indicator
        typingIndicator.remove();

        chatMessages.appendChild(createMessageElement(
            'Maaf, terjadi kesalahan dalam koneksi. Silakan coba lagi.'
        ));
    }

    scrollToBottom();
}

// Enable form submission on Enter key
document.getElementById('userInput').addEventListener('keypress', function(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        document.getElementById('chatForm').dispatchEvent(new Event('submit'));
    }
});

