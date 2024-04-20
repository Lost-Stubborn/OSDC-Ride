// Create a WebSocket connection to the server using the native WebSocket API
const ws = new WebSocket('ws://localhost:8765/');

// Event handler for when the WebSocket connection is established
ws.onopen = () => {
    console.log('WebSocket client connected');
};

// Event handler for incoming messages from the WebSocket server
ws.onmessage = (event) => {
    console.log(`Received message: ${event.data}`);
    // Handle response from server if needed
};

document.getElementById('click_to_convert').addEventListener('click', function() {
    var speech = true;
    window.SpeechRecognition = window.webkitSpeechRecognition || window.SpeechRecognition;
    const recognition = new SpeechRecognition();
    recognition.interimResults = true;

    recognition.addEventListener('result', e => {
        const transcript = Array.from(e.results)
            .map(result => result[0])
            .map(result => result.transcript)
            .join(' '); // Combine all speech pieces into a single string

        document.getElementById('convert_text').value = transcript;

        // Send transcript data to the Flask server via POST request
        fetch('/submit', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ transcript: transcript })
        }).then(response => response.json())
          .then(data => console.log(data))
          .catch(error => console.error('Error:', error));
    });

    if (speech) {
        recognition.start();
    }
});


