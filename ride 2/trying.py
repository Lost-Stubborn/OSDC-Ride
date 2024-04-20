import chatboypython as c
from flask import Flask, request, redirect, render_template
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

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
    with open('templates\info\data.json', 'w') as json_file:
        c.json.dump(result_data, json_file)
    # process_in_jupyter(data)
    return redirect('/')

def process_in_jupyter(data):
    # Here, add code to invoke Jupyter notebook processing
    print(f"Data received: {data}")

if __name__ == '__main__':
    app.run(debug=False)


# while True:
#     user_input = input("User: ")
#     intent = c.predict_intent(user_input)
#     response = c.generate_response(intent)
#     data = {'message': response}
#     with open('data.json', 'w') as json_file:
#         c.json.dump(data, json_file)
#     print("Chatbot:", response)

