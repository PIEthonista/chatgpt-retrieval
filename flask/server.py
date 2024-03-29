from flask import Flask, render_template, request, jsonify, redirect, session, url_for
from flask_cors import CORS, cross_origin
import pickle
import joblib

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://127.0.0.1:5000", "*"]}})

@app.route('/')
def home():
    return 'Hello World'

@app.route('/model_output', methods=['POST'])
def record_user_expenses():
    # Get the JSON data from the request
    model_output = request.json.get('model_output')
    # This print statement should be saving to database
    print("User expenses received:", model_output)
    return jsonify(model_output)

@app.route('/get_chat_history')
def get_chat_history():
    # Get the key from javascript FormData object
    # Example in service.js of frontend
    # const anomalyForm = new FormData();
    # anomalyForm.append("paragraph", transcription);
    output = request.form.get('response')

@app.route('/get_data_from_database', methods=['GET'])
def get_data_from_database():
    # Get the key from javascript FormData object
    # Example in service.js of frontend
    # const anomalyForm = new FormData();
    # anomalyForm.append("paragraph", transcription);
    output = request.form.get('response')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')