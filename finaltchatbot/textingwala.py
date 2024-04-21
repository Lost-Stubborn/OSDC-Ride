import chatboypython as c
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('chatbot.html')

@app.route('/submit', methods=['POST'])
def submit_data():
    data = request.get_json()
    transcript = data['transcript']  # Get the transcript from the JSON data
    intent = c.predict_intent(transcript)
    response = c.generate_response(intent)
    result_data = {'message': response}
    
    return jsonify(result_data)  # Send JSON response back to the client

if __name__ == '__main__':
    app.run(debug=False)
