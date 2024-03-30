from flask import Flask, render_template, request, jsonify, redirect, session, url_for
from flask_cors import CORS, cross_origin
import pickle
import joblib
import json
from openai import OpenAI
from api_keys import OPENAI_API_KEY
from main import init_record_user_expense_income_model, init_functional_model, action_layer, __DEBUGGING__, __MAX_ERROR_TRIAL__, record_user_expenses_income, expenses_prediction, data_analysis, regular_chat
from api_keys import OPENAI_API_KEY, TAVILY_API_KEY
from system_prompts import AUTOMATED_EXPENSE_RECORDING_SP, DATA_ANALYSIS_SP, EXPENSE_PREDICTION_SP, REGULAR_CHAT_SP, \
                           PERSONA_GUARDIAN_SP, PERSONA_TEACHER_SP, PERSONA_ADVISOR_SP, \
                           MODEL_POWERS_SP
from func_call_template import MANDATORY_FUNCTION_CALL_TEMPLATE, FUNCTION_CALL_TEMPLATE
from model_config import __MODEL__, __MAX_TOKENS__, __TEMPERATURE__, __VERBOSE__, __USER_DATA_PATH__, __MAX_DATE_RANGE__, \
                         __DEBUGGING__, __MAX_ERROR_TRIAL__, \
                         __PERSONA_TEACHER_AGE__, __PERSONA_GUARDIAN_AGE__, __PERSONA_ADVISOR_AGE__
import sys
from langchain_core.messages import HumanMessage, AIMessage
import re

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://172.20.10.4:5000", "*"]}})

def is_user(history):
    user = history['user']
    if user['_id'] == 1:
        return False
    else:
        return True
    
def convert_json(prompt):
    pattern = r"'(\w+)'\s*:\s*(\d+)"
    matches = re.findall(pattern, prompt)
    kv_pairs = {key: int(value) for key, value in matches}
    json_data = json.dumps(kv_pairs, indent=2)
    return json_data
    
def use_model(prompt, chat_history, user_details):
    new_chat_history = []

    if chat_history != []:
        for history in chat_history:
            if is_user(history):
                new_chat_history.append(HumanMessage(content=history['text']))
            else:
                new_chat_history.append(AIMessage(content=history['text']))

    user_name = user_details['username']
    user_age = user_details['age']
    user_mbti = user_details['mbti']
    user_gender = user_details['gender']
    user_address = user_details['address']
    user_set_model_name = user_details['model_name']
    
    # preset persona
    if user_age in range(*__PERSONA_TEACHER_AGE__):
        persona = PERSONA_TEACHER_SP
    elif user_age in range(*__PERSONA_GUARDIAN_AGE__):
        persona = PERSONA_GUARDIAN_SP
    else:
        persona = PERSONA_ADVISOR_SP
    
    persona = f"Your name is {user_set_model_name}. " + persona + \
        f" The end user's personal info is as below. Name {user_name}, Age {user_age}, MBTI {user_mbti}, \
        Gender {user_gender} and is currently living at {user_address}. \
        Please customise and personalise your responses for the end user based on those."
    
    # init required models
    client = OpenAI(api_key=OPENAI_API_KEY)
    model_record_user_expense = init_record_user_expense_income_model()
    model_regular_chat = init_functional_model(REGULAR_CHAT_SP + " " + MODEL_POWERS_SP + " " + persona)
    model_data_analysis = init_functional_model(DATA_ANALYSIS_SP + " " + MODEL_POWERS_SP + " " + persona)
    model_expenses_prediction = init_functional_model(EXPENSE_PREDICTION_SP + " " + MODEL_POWERS_SP + " " + persona)
        
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

    response_type = ''
    data = ''
    # function calls must be in order
    if 'record_user_expenses' in funcs_to_call.keys():
        output = None
        c = 0
        while (output is None) and (c < __MAX_ERROR_TRIAL__):
            output = record_user_expenses_income(agent_executor=model_record_user_expense, user_input=prompt)
            c += 1
        print("\n--- record_user_expenses()") if __DEBUGGING__ else None
        print(output)
        if output is not None:
            formatted_output = "I recorded your expense/income as follows: " + output
        else:
            formatted_output = ''
        new_chat_history.append(AIMessage(content=formatted_output))
        final_output = formatted_output
        response_type = 'record_expenses'
        data = convert_json(output)
        
    if 'expense_prediction' in funcs_to_call.keys():
        output = None
        c = 0
        while (output is None) and (c < __MAX_ERROR_TRIAL__):
            output = expenses_prediction(client=client, agent_executor=model_expenses_prediction, chat_history=new_chat_history, user_input=prompt)
            c += 1
        print("\n--- expenses_prediction()") if __DEBUGGING__ else None
        print(output)
        if output is not None:
            new_chat_history.append(AIMessage(content=output))
        else:
            new_chat_history.append(AIMessage(content=''))
        final_output = output
        response_type = 'regular_answer'

    if 'data_analysis' in funcs_to_call.keys():
        output = None
        c = 0
        while (output is None) and (c < __MAX_ERROR_TRIAL__):
            output = data_analysis(client=client, agent_executor=model_data_analysis, chat_history=new_chat_history, user_input=prompt)
            c += 1
        print("\n--- data_analysis()") if __DEBUGGING__ else None
        print(output)
        if output is not None:
            new_chat_history.append(AIMessage(content=output))
        else:
            new_chat_history.append(AIMessage(content=''))
        final_output = output
        response_type = 'regular_answer'

    # final revert
    if 'regular_chat' in funcs_to_call.keys():
        output = None
        c = 0
        while (output is None) and (c < __MAX_ERROR_TRIAL__):
            output = regular_chat(agent_executor=model_regular_chat, chat_history=new_chat_history, user_input=prompt)
            c += 1
        print("\n--- regular_chat()") if __DEBUGGING__ else None
        print(output)
        if output is not None:
            new_chat_history.append(AIMessage(content=output))
        else:
            new_chat_history.append(AIMessage(content=''))
        final_output = output
        response_type = 'regular_answer'

    return {'response_type': response_type, 'data':data, 'content': final_output}

@app.route('/')
def home():
    return 'Hello World'

@app.route('/get_user_response', methods=['POST'])
def get_user_response():
    user_input = request.form.get('message_to_gpt')
    chat_history = json.loads(request.form.get('conversation_history'))
    user_details = json.loads(request.form.get('user_details'))
    print("DATAAAA: ",user_input)
    updated_result = use_model(user_input, chat_history, user_details)
    return jsonify(updated_result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')