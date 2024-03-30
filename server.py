from flask import Flask, render_template, request, jsonify, redirect, session, url_for
from flask_cors import CORS, cross_origin
import json
import pickle
import joblib
from openai import OpenAI
from api_keys import OPENAI_API_KEY
from main import init_record_user_expense_income_model, init_regular_chat_model, init_data_analysis_model, init_expenses_prediction_model, action_layer, __DEBUGGING__, __MAX_ERROR_TRIAL__, record_user_expenses_income, expenses_prediction, data_analysis, regular_chat
import sys
from langchain_core.messages import HumanMessage, AIMessage

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://172.20.10.4:5000", "*"]}})

def is_user(history):
    user = history['user']
    if user['_id'] == 1:
        return False
    else:
        return True

def use_model(prompt, chat_history):
    new_chat_history = []

    if chat_history != []:
        for history in chat_history:
            if is_user(history):
                new_chat_history.append(HumanMessage(content=history['text']))
            else:
                new_chat_history.append(AIMessage(content=history['text']))

    client = OpenAI(api_key=OPENAI_API_KEY)
    model_record_user_expense = init_record_user_expense_income_model()
    model_regular_chat = init_regular_chat_model()
    model_data_analysis = init_data_analysis_model()
    model_expenses_prediction = init_expenses_prediction_model()
    
    # append to chat history
    new_chat_history.append(HumanMessage(content=prompt))
    # action layer
    completion = action_layer(client=client, user_input=prompt)
    # print(completion)
    
    funcs_to_call = {}
    if completion.choices[0].message.tool_calls is not None:
        for i in completion.choices[0].message.tool_calls:
            funcs_to_call[i.function.name] = i.function.arguments
            # funcs_to_call[i.message.function_call.name] = i.message.function_call.arguments
                # funcs_to_call[i.message.tool_calls]
    else:
        funcs_to_call['regular_chat'] = ""
        
    # debugging
    if __DEBUGGING__:
        for k,v in funcs_to_call.items():
            print(k, "-> ", end='', flush=True)
        print()


    # function calls must be in order
    if 'record_user_expenses' in funcs_to_call.keys():
        output = None
        c = 0
        while (output is None) and (c < __MAX_ERROR_TRIAL__):
            output = record_user_expenses_income(agent_executor=model_record_user_expense, user_input=prompt)
            c += 1
        print("\n--- record_user_expenses()") if __DEBUGGING__ else None
        print(output)
        formatted_output = "I recorded your expense/income as follows: " + output
        if formatted_output is not None:
            new_chat_history.append(AIMessage(content=formatted_output))
        else:
            new_chat_history.append(AIMessage(content=''))
        # must let model know that it recorded the expense
        
    if 'expense_prediction' in funcs_to_call.keys():
        output = None
        c = 0
        while (output is None) and (c < __MAX_ERROR_TRIAL__):
            output = expenses_prediction(client=client, agent_executor=model_expenses_prediction, chat_history=chat_history, user_input=prompt)
            c += 1
        print("\n--- expenses_prediction()") if __DEBUGGING__ else None
        print(output)
        if output is not None:
            new_chat_history.append(AIMessage(content=output))
        else:
            new_chat_history.append(AIMessage(content=''))

    if 'data_analysis' in funcs_to_call.keys():
        output = None
        c = 0
        while (output is None) and (c < __MAX_ERROR_TRIAL__):
            output = data_analysis(client=client, agent_executor=model_data_analysis, chat_history=chat_history, user_input=prompt)
            c += 1
        print("\n--- data_analysis()") if __DEBUGGING__ else None
        print(output)
        if output is not None:
            new_chat_history.append(AIMessage(content=output))
        else:
            new_chat_history.append(AIMessage(content=''))
        

    # final revert
    if 'regular_chat' in funcs_to_call.keys():
        output = None
        c = 0
        while (output is None) and (c < __MAX_ERROR_TRIAL__):
            output = regular_chat(agent_executor=model_regular_chat, chat_history=chat_history, user_input=prompt)
            c += 1
        print("\n--- regular_chat()") if __DEBUGGING__ else None
        print(output)
        if output is not None:
            new_chat_history.append(AIMessage(content=output))
        else:
            new_chat_history.append(AIMessage(content=''))

    return output

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

@app.route('/get_user_response', methods=['POST'])
def get_user_response():
    user_input = request.form.get('message_to_gpt')
    chat_history = json.loads(request.form.get('conversation_history'))

#     Example of chat history
#     const [messages, setMessages] = useState([
#     {
#       text: "Hey my finance bot, how are you",
#       user: { _id: 1, name: `${authenticationState.username}` },
#       createdAt: Timestamp.now() + 1,
#     },
#     {
#       text:
#         "I am fine " +
#         `${authenticationState.username}. How can I help you today?`,
#       user: { _id: 2, name: `Chat Bot` },
#       createdAt: Timestamp.now() + 2,
#     },
#     {
#       text: "bruhhhhh",
#       user: { _id: 1, name: `${authenticationState.username}` },
#       createdAt: Timestamp.now() + 3,
#     },
#   ]);
    print("DATAAAA: ",user_input)
    updated_result = use_model(user_input, chat_history)
    response = {'result': updated_result}
    return jsonify(response)

@app.route('/get_data_from_database', methods=['GET'])
def get_data_from_database():
    # Get the key from javascript FormData object
    # Example in service.js of frontend
    # const anomalyForm = new FormData();
    # anomalyForm.append("paragraph", transcription);
    output = request.form.get('response')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')