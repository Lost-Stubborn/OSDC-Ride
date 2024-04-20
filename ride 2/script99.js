let lastMessage = ''; // Variable to store the last spoken message

// Function to speak the message
function speakMessage(message) {
    const speech = new SpeechSynthesisUtterance(message);
    window.speechSynthesis.speak(speech);
}

// Function to fetch and process data
function fetchDataAndSpeak() {
    // Send a request to retrieve the JSON data from the server
    fetch('data.json')
        .then(response => response.json())
        .then(jsonData => {
            // Process the received JSON data
            console.log('Received data from Python:', jsonData);

            // Extract the message from the JSON data
            const message = jsonData.message;

            // Check if the new message is different from the last spoken message
            if (message !== lastMessage) {
                // Speak the message using the Web Speech API
                speakMessage(message);

                // Update the last spoken message
                lastMessage = message;
            }
        })
        .catch(error => {
            console.error('Error fetching JSON data:', error);
        });
}

// Set an interval to periodically fetch and process data
const intervalId = setInterval(fetchDataAndSpeak, 5000); // Adjust the interval as needed (in milliseconds)

// Stop the interval after a certain period (e.g., 5 minutes)
setTimeout(() => {
    clearInterval(intervalId);
}, 300000); // 300000 milliseconds = 5 minutes
