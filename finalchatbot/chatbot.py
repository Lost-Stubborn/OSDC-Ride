import chatboypython as c
from flask import Flask, request, jsonify, render_template, redirect
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index2')
def videopage():
    return render_template('index2.html')

@app.route('/chatbot')
def chattingpage():
    return render_template('chatbot.html')

@app.route('/submit', methods=['POST'])
def submit_data():
    # Receive JSON data sent from the frontend
    data = request.get_json()
    transcript = data['transcript']  # Get the transcript from the JSON data
    # Process the transcript data with your chatbot logic or any other processing
    user_input = transcript
    intent = c.predict_intent(user_input)
    response = c.generate_response(intent)
    result_data = {'message': response}
    with open('static\data.json', 'w') as json_file:
        c.json.dump(result_data, json_file)



    data = {'intent': intent, 'frequency': 1}  # Set frequency to 1 for each entry

    # Write data to a JSON file (result.json)
    with open('result.json', 'a') as json_file:
        json.dump(data, json_file)
        json_file.write('\n')  # Add a newline character to separate entries


    

    # process_in_jupyter(data)
    return redirect('/')



@app.route('/submit2', methods=['POST'])
def submit_data2():
    data = request.get_json()
    transcript = data['transcript']  # Get the transcript from the JSON data
    intent = c.predict_intent(transcript)
    response = c.generate_response(intent)
    result_data = {'message': response}
    
    return jsonify(result_data)  # Send JSON response back to the client

@app.route('/data')
def datanalys():
    return render_template('data.html')

# @app.route('/login')
# def login():
#     return render_template('login.html')

if __name__ == '__main__':
    app.run(debug=False)

