import os
import sys
from openai import OpenAI
import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_openai_functions_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor


from constants import OPENAI_API_KEY, TAVILY_API_KEY
from system_prompts import AUTOMATED_EXPENSE_RECORDING

__VERBOSE__ = False
__MODEL__ = "gpt-3.5-turbo-1106"




function_call_template = [
    {
        "type": "function",
        "function": {
            'name': 'get_current_temperature',
            'description': 'gets the current temperature',
            'parameters': {}
        }
    },
    {
        "type": "function",
        "function": {
            'name': 'get_current_traffic',
            'description': 'gets the current traffic',
            'parameters': {}
        }
    },
    {
        "type": "function",
        "function": {
            'name': 'record_user_expenses',
            'description': 'extract expenses information and categorise them, along with their values',
            'parameters': {}
        }
    },
    {
        "type": "function",
        "function": {
            'name': 'acquire_history_data',
            'description': 'obtains user past and current expenses data from start_date to end_date. Please call at least',
            'parameters': {
                'type': 'object',
                'properties': {
                    'start_date': {
                        'type': 'string',
                        'description': 'start date in the format of DD-MM-YYYY, for user expense data retrieval'
                    },
                    'end_date': {
                        'type': 'string',
                        'description': 'end date in the format of DD-MM-YYYY, for user expense data retrieval'
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            'name': 'expense_prediction',
            'description': 'predicts user future expenses based on past expenses data. If this function needs to be called, please call the acquire_history_data() function beforehand.',
            'parameters': {}
        }
    },
    {
        "type": "function",
        "function": {
            'name': 'data_analysis',
            'description': 'call this function when user asks about anything related to / possibly related to the users past data. If this function needs to be called, please call the acquire_history_data() function beforehand.',
            'parameters': {}
        }
    },
    {
        "type": "function",
        "function": {
            'name': 'regular_chat',
            'description': 'must revert to calling this function when user input is not related to any other given functions',
            'parameters': {}
        }
    }
]

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
    # function_choice = client.chat.completions.create(
    #     model = __MODEL__,
    #     messages = [{"role": "user", "content": user_input}],
    #     functions = function_call_template,   # <========add this parameter
    #     function_call = 'auto',     # <========add this parameter
    #     max_tokens = 1024,
    #     temperature = 0
    # )
    function_choice = client.chat.completions.create(
        model = __MODEL__,
        messages = [{"role": "user", "content": user_input}],
        tools = function_call_template,   # <========add this parameter
        tool_choice = 'auto',     # <========add this parameter
        max_tokens = 1024,
        temperature = 0
    )
    return function_choice # .choices[0].message.content.strip()


def init_record_user_experience_model():
    llm_record = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=__MODEL__)
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", AUTOMATED_EXPENSE_RECORDING),
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

def record_user_expenses(agent_executor:AgentExecutor, user_input):
    chat_history = []  # not required here
    response = agent_executor.invoke({
        "chat_history": chat_history,
        "input": user_input
    })
    output = response['output']
    return output



def init_regular_chat_model():
    llm_chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=__MODEL__)
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "you are a helpful asistant and you are to only talk about finance or financial literacy"),
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

    response = agent_executor.invoke({
        "chat_history": chat_history,
        "input": user_input
    })
    output = response['output']
    return output
    



# print(completion.choices[0].message.function_call.name)
# print(completion.choices[0].message.function_call.arguments)

# print(json.loads(chat_gpt(prompt)))


if __name__ == "__main__":
    # init required models
    client = OpenAI(api_key=OPENAI_API_KEY)
    model_record_user_experience = init_record_user_experience_model()
    model_regular_chat = init_regular_chat_model()
    
    chat_history = []
    while True:
        prompt = input(">>> Prompt: ")
        if prompt in ['quit', 'q', 'exit']:
            sys.exit()
        
        # append to chat history
        chat_history.append(HumanMessage(content=prompt))
        
        # action layer
        completion = action_layer(client=client, user_input=prompt)
        # print(completion)
        
        funcs_to_call = {}
        user_history_data = None
        for i in completion.choices[0].message.tool_calls:
            funcs_to_call[i.function.name] = i.function.arguments
            # funcs_to_call[i.message.function_call.name] = i.message.function_call.arguments
                # funcs_to_call[i.message.tool_calls]
            
        # debugging
        for k,v in funcs_to_call.items():
            print(k, "-> ", end='', flush=True)
        print()


        # function calls must be in order (acquire_history_data before any function that needs history data)
        if 'record_user_expenses' in funcs_to_call.keys():
            output = record_user_expenses(agent_executor=model_record_user_experience, user_input=prompt)
            print("\nrecord_user_expenses()")
            print(output)
            # dont record output to chat_history
            
        if 'acquire_history_data' in funcs_to_call.keys():
            # TODO: implement method//
            # user_history_data = ...
            print("\nacquire_history_data()")
            # dont record output to chat_history

        if 'expense_prediction' in funcs_to_call.keys():
            if user_history_data is not None:
                # TODO: implement method//
                # chat_history.append(AIMessage(content=output))
                pass
            else:
                print("\nexpense_prediction()")
                mock_output = "based on your past data, you are going to go broke :D"
                chat_history.append(AIMessage(content=mock_output))
        
        if 'data_analysis' in funcs_to_call.keys():
            if user_history_data is not None:
                # TODO: implement method//
                # chat_history.append(AIMessage(content=output))
                pass
            else:
                print("\ndata_analysis()")
                mock_output = "based on your past data, you spent too much.."
                chat_history.append(AIMessage(content=mock_output))
        
        # final revert
        if 'regular_chat' in funcs_to_call.keys():
            output = regular_chat(agent_executor=model_regular_chat, chat_history=chat_history, user_input=prompt)
            print("\nregular_chat()")
            print(output)
            chat_history.append(AIMessage(content=output))
        
        
        
        

