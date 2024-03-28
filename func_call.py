import os
import openai
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

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY


my_functions = [
    {
        'name': 'record_user_expenses',
        'description': 'extract expenses information and categorise them, along with their values',
        'parameters': {}
    },
    {
        'name': 'acquire_history_data',
        'description': 'obtains user past and current expenses data from start_date to end_date',
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
    },
    {
        'name': 'expense_prediction',
        'description': 'predicts user future expenses based on past expenses data',
        'parameters': {}
    },
    {
        'name': 'data_analysis',
        'description': 'performs data analysis to obtain user required statistic or answer on past user expenses data',
        'parameters': {}
    },
    {
        'name': 'regular_chat',
        'description': 'must call this function when user input is not related to any other given functions',
        'parameters': {}
    },
]

prompt = 'i had a meal at Mac D today. Had a fish fillet for RM7.90, oh and another ice cream for 1. i also went shopping and clubbing with my friends after that. I think i spent around 75 overall for that'
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


client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=OPENAI_API_KEY,
)

def chat_gpt(user_input):
    response = client.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages = [{"role": "user", "content": user_input}],
        functions = my_functions,   # <========add this parameter
        function_call = 'auto',     # <========add this parameter
        max_tokens = 1024,
        temperature = 0
    )
    return response # .choices[0].message.content.strip()

completion = chat_gpt(prompt)
print(completion)
funcs_to_call = []
for i in completion.choices:
    print(i.message.function_call.name)
    funcs_to_call.append(i.message.function_call.name)


def record_user_expenses(user_input):
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
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
    chat_history = [HumanMessage(content=user_input)]
    agent = create_openai_functions_agent(llm, tools, prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=__VERBOSE__)
    response = agent_executor.invoke({
        "chat_history": chat_history,
        "input": user_input
    })
    output = response['output']
    return output
    


if 'record_user_expenses' in funcs_to_call:
    output = record_user_expenses(prompt)
    print("===========")
    print(output)

# print(completion.choices[0].message.function_call.name)
# print(completion.choices[0].message.function_call.arguments)

# print(json.loads(chat_gpt(prompt)))


