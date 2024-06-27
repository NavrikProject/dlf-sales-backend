
from flask import Flask, request, jsonify
from flask_cors import CORS
from LangchainConnection import invoke_chain
import json
app = Flask(__name__)
CORS(app)

# Data Template
'''
#Template
'': {
        'message': [""],
        'options': [],
    }
'''

# Home route API for testing


@app.route('/')
def home():
    return 'Hello, world!'

# API to fetch Initial menu options

@app.route('/process_message', methods=['POST'])
def process_message():
    # Get message from the app
    userQuestion = request.json.get('message')
    
    messages = []
    # Process the message using NLP processor
    response = invoke_chain(userQuestion,messages)
    if response == "It seems like the user question provided is not clear or relevant to the SQL query and result provided. Please provide a more specific or relevant question for me to answer.":
        return jsonify({'response': "Please provide a more specific question for me to answer"})
    else:
        # Return the response to the app
        return jsonify({'response': response})


if __name__ == "__main__":
    app.run(debug=True)
