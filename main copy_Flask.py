import os
import sys
import json
import datetime

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_openai_functions_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor


from api_keys import OPENAI_API_KEY, TAVILY_API_KEY
from system_prompts import AUTOMATED_EXPENSE_RECORDING_SP, DATA_ANALYSIS_SP, EXPENSE_PREDICTION_SP, REGULAR_CHAT_SP
from func_call_template import MANDATORY_FUNCTION_CALL_TEMPLATE, FUNCTION_CALL_TEMPLATE
from model_config import __MODEL__, __MAX_TOKENS__, __TEMPERATURE__, __VERBOSE__, __USER_DATA_PATH__
from utils import get_csv_given_date

import requests


# prompt = 'i had a meal at Mac D today. Had a fish fillet for RM7.90, oh and another ice cream for 1. i also went shopping and clubbing with my friends after that. I think i spent around 75 overall for that'
# prompt = '''
# Cause we were just kids when we fell in love
# Not knowing what it was
# I will not give you up this time
# But darling, just kiss me slow
# Your heart is all I own
# And in your eyes, you're holding mine
# Baby, I'm dancing in the dark
# With you between my arms
# '''
# prompt = "Hi who is the current sultan of malaysia?"




def action_layer(client: OpenAI, user_input):
    #     functions = function_call_template,   # <========add this parameter
    #     function_call = 'auto',     # <========add this parameter
    function_choice = client.chat.completions.create(
        model = __MODEL__,
        messages = [{"role": "user", "content": user_input}],
        tools = FUNCTION_CALL_TEMPLATE,
        tool_choice = 'auto',
        max_tokens = __MAX_TOKENS__,
        temperature = __TEMPERATURE__
    )
    return function_choice


def get_start_end_date_for_history(client: OpenAI, user_input):
    formatted_input = "The current datetime is " + datetime.datetime.now().strftime("%A, %B %d, %Y %H:%M:%S") + ". " + user_input
    completion = client.chat.completions.create(
        model = __MODEL__,
        messages = [{"role": "user", "content": formatted_input}],
        tools = MANDATORY_FUNCTION_CALL_TEMPLATE,
        tool_choice = {"type": "function", "function": {"name": "acquire_history_data"}},     # force model to call this tool // mandatory
        max_tokens = __MAX_TOKENS__,
        temperature = __TEMPERATURE__
    )
    
    # if no date specified
    if not completion.choices[0].message.tool_calls[0].function.name and not completion.choices[0].message.tool_calls[0].function.arguments:
        current_date = datetime.datetime.now()
        one_month_ago = current_date - datetime.timedelta(days=30)
        formatted_current_date = current_date.strftime("%d-%m-%Y")
        formatted_one_month_ago = one_month_ago.strftime("%d-%m-%Y")
        print("--- calculating date")
        return {
                    "start_date": formatted_one_month_ago, 
                    "end_date": formatted_current_date
               }
    else:
        print("--- model return date")
        return json.loads(completion.choices[0].message.tool_calls[0].function.arguments)



def init_record_user_expense_income_model():
    llm_record = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=__MODEL__)
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", AUTOMATED_EXPENSE_RECORDING_SP),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
    search = TavilySearchResults()
    tools = [search]      # must at least has a tool for agent. solution: dont use agent?
    agent = create_openai_functions_agent(llm_record, tools, prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=__VERBOSE__)
    return agent_executor


def record_user_expenses_income(agent_executor:AgentExecutor, user_input):
    formatted_input = "The current datetime is " + datetime.datetime.now().strftime("%A, %B %d, %Y %H:%M:%S") + ". " + user_input
    chat_history = []  # not required here
    response = agent_executor.invoke({
        "chat_history": chat_history,
        "input": formatted_input
    })
    output = response['output']
    return output



def init_regular_chat_model():
    llm_chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=__MODEL__)
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", REGULAR_CHAT_SP),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
    search = TavilySearchResults()
    tools = [search]      # must at least has a tool for agent. solution: dont use agent?
    agent = create_openai_functions_agent(llm_chat, tools, prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=__VERBOSE__)
    return agent_executor
    

def regular_chat(agent_executor: AgentExecutor, chat_history, user_input):
    formatted_input = "The current datetime is " + datetime.datetime.now().strftime("%A, %B %d, %Y %H:%M:%S") + ". " + user_input
    response = agent_executor.invoke({
        "chat_history": chat_history,
        "input": formatted_input
    })
    output = response['output']
    return output


def init_data_analysis_model():
    llm_data_analysis = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=__MODEL__)
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", DATA_ANALYSIS_SP),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
    search = TavilySearchResults()
    tools = [search]      # must at least has a tool for agent. solution: dont use agent?
    agent = create_openai_functions_agent(llm_data_analysis, tools, prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=__VERBOSE__)
    return agent_executor


def data_analysis(client: OpenAI, agent_executor: AgentExecutor, chat_history, user_input):
    # call acquire history data
    date_dict = get_start_end_date_for_history(client=client, user_input=user_input)
    print("--- selected date", date_dict)
    data_csv = get_csv_given_date(csv_path=__USER_DATA_PATH__, start_date=date_dict['start_date'], end_date=date_dict['end_date'])
    print("--- data_csv\n", data_csv)
    
    contexted_input = "The current datetime is " + datetime.datetime.now().strftime("%A, %B %d, %Y %H:%M:%S") + ". " \
                      + user_input + "User's past expense data are as follows: " + data_csv
    response = agent_executor.invoke({
        "chat_history": chat_history,
        "input": contexted_input
    })
    output = response['output']
    return output


def init_expenses_prediction_model():
    llm_expense_prediction = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=__MODEL__)
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", EXPENSE_PREDICTION_SP),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
    search = TavilySearchResults()
    tools = [search]      # must at least has a tool for agent. solution: dont use agent?
    agent = create_openai_functions_agent(llm_expense_prediction, tools, prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=__VERBOSE__)
    return agent_executor


def expenses_prediction(client: OpenAI, agent_executor: AgentExecutor, chat_history, user_input):
    # call acquire history data
    date_dict = get_start_end_date_for_history(client=client, user_input=user_input)
    print("--- selected date", date_dict)
    data_csv = get_csv_given_date(csv_path=__USER_DATA_PATH__, start_date=date_dict['start_date'], end_date=date_dict['end_date'])
    print("--- data_csv\n", data_csv)
    
    contexted_input = "The current datetime is " + datetime.datetime.now().strftime("%A, %B %d, %Y %H:%M:%S") + ". " \
                      + user_input + "User's past expense data are as follows: " + data_csv
    response = agent_executor.invoke({
        "chat_history": chat_history,
        "input": contexted_input
    })
    output = response['output']
    return output


# print(completion.choices[0].message.function_call.name)
# print(completion.choices[0].message.function_call.arguments)

# print(json.loads(chat_gpt(prompt)))

if __name__ == "__main__":
    # init required models
    client = OpenAI(api_key=OPENAI_API_KEY)
    model_record_user_expense = init_record_user_expense_income_model()
    model_regular_chat = init_regular_chat_model()
    model_data_analysis = init_data_analysis_model()
    model_expenses_prediction = init_expenses_prediction_model()
    
    chat_history = []
    # while True:
    prompt = input(">>> Prompt: ")
    if prompt in ['quit', 'q', 'exit']:
        sys.exit()
    
    # append to chat history
    chat_history.append(HumanMessage(content=prompt))
    
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
    for k,v in funcs_to_call.items():
        print(k, "-> ", end='', flush=True)
    print()


    # function calls must be in order (acquire_history_data before any function that needs history data)
    if 'record_user_expenses' in funcs_to_call.keys():
        output = record_user_expenses_income(agent_executor=model_record_user_expense, user_input=prompt)
        print("\n--- record_user_expenses()")
        print(output)
        formatted_output = "I recorded your expense/income as follows: " + output
        chat_history.append(AIMessage(content=formatted_output))
        # dont record output to chat_history
        
    if 'expense_prediction' in funcs_to_call.keys():
        output = expenses_prediction(client=client, agent_executor=model_expenses_prediction, chat_history=chat_history, user_input=prompt)
        print("\n--- expenses_prediction()")
        print(output)
        chat_history.append(AIMessage(content=output))

    if 'data_analysis' in funcs_to_call.keys():
        output = data_analysis(client=client, agent_executor=model_data_analysis, chat_history=chat_history, user_input=prompt)
        print("\n--- data_analysis()")
        print(output)
        chat_history.append(AIMessage(content=output))

    # final revert
    if 'regular_chat' in funcs_to_call.keys():
        output = regular_chat(agent_executor=model_regular_chat, chat_history=chat_history, user_input=prompt)
        print("\n--- regular_chat()")
        print(output)
        chat_history.append(AIMessage(content=output))
    
    # send output to database to store
    response = requests.post('http://127.0.0.1:5000/model_output', headers={'Content-type':'application/json', 'Accept':'application/json'
}, json={'model_output': output})
    if response.status_code != 200:
        print("Failed to update chat history on server.")
        
        
        
        

