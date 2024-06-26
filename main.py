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

from langchain.chains import LLMMathChain 
from langchain.agents import Tool


from api_keys import OPENAI_API_KEY, TAVILY_API_KEY
from system_prompts import AUTOMATED_EXPENSE_RECORDING_SP, DATA_ANALYSIS_SP, EXPENSE_PREDICTION_SP, REGULAR_CHAT_SP, \
                           PERSONA_GUARDIAN_SP, PERSONA_TEACHER_SP, PERSONA_ADVISOR_SP, \
                           MODEL_POWERS_SP
from func_call_template import MANDATORY_FUNCTION_CALL_TEMPLATE, FUNCTION_CALL_TEMPLATE
from model_config import __MODEL__, __MAX_TOKENS__, __TEMPERATURE__, __VERBOSE__, __USER_DATA_PATH__, __MAX_DATE_RANGE__, \
                         __DEBUGGING__, __MAX_ERROR_TRIAL__, \
                         __PERSONA_TEACHER_AGE__, __PERSONA_GUARDIAN_AGE__, __PERSONA_ADVISOR_AGE__
from utils import get_csv_given_date

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
        one_month_ago = current_date - datetime.timedelta(days=__MAX_DATE_RANGE__)
        formatted_current_date = current_date.strftime("%d-%m-%Y")
        formatted_one_month_ago = one_month_ago.strftime("%d-%m-%Y")
        print("--- calculating date") if __DEBUGGING__ else None
        return {
                    "start_date": formatted_one_month_ago, 
                    "end_date": formatted_current_date
               }
    else:
        print("--- model return date") if __DEBUGGING__ else None
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
    
    problem_chain = LLMMathChain.from_llm(llm=llm_record)
    math_tool = Tool.from_function(name="Calculator",
                    func=problem_chain.run,
                    description="Useful for when you need to answer questions about math. This tool is only for math questions and nothing else. Only input math expressions. Must use this tool whenever end user prompts anythign related to math.")
    
    tools = [search, math_tool]      # must at least has a tool for agent. solution: dont use agent?
    agent = create_openai_functions_agent(llm_record, tools, prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=__VERBOSE__)
    return agent_executor

def init_summarize_text_model():
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
    
    problem_chain = LLMMathChain.from_llm(llm=llm_record)
    math_tool = Tool.from_function(name="Calculator",
                    func=problem_chain.run,
                    description="Useful for when you need to answer questions about math. This tool is only for math questions and nothing else. Only input math expressions. Must use this tool whenever end user prompts anythign related to math.")
    
    tools = [search, math_tool]      # must at least has a tool for agent. solution: dont use agent?
    agent = create_openai_functions_agent(llm_record, tools, prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=__VERBOSE__)
    return agent_executor

def record_user_expenses_income(agent_executor:AgentExecutor, user_input):
    formatted_input = "The current datetime is " + datetime.datetime.now().strftime("%A, %B %d, %Y %H:%M:%S") + ". " + user_input
    chat_history = []  # not required here
    try:
        response = agent_executor.invoke({
            "chat_history": chat_history,
            "input": formatted_input
        })
        output = response['output']
        return output
    except:
        return None


def init_functional_model(system_prompt):
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=__MODEL__)
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
    search = TavilySearchResults()
    
    problem_chain = LLMMathChain.from_llm(llm=llm)
    math_tool = Tool.from_function(name="Calculator",
                    func=problem_chain.run,
                    description="Useful for when you need to answer questions about math. This tool is only for math questions and nothing else. Only input math expressions. Must use this tool whenever end user prompts anythign related to math.")

    tools = [search, math_tool]      # must at least has a tool for agent. solution: dont use agent?
    agent = create_openai_functions_agent(llm, tools, prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=__VERBOSE__)
    return agent_executor

def regular_chat(agent_executor: AgentExecutor, chat_history, user_input):
    formatted_input = "The current datetime is " + datetime.datetime.now().strftime("%A, %B %d, %Y %H:%M:%S") + ". " + user_input
    try:
        response = agent_executor.invoke({
            "chat_history": chat_history,
            "input": formatted_input
        })
        output = response['output']
        return output
    except:
        return None

def summarize_text(client: OpenAI, agent_executor: AgentExecutor, chat_history, user_input):
    formatted_input = "Just assume you are talking to me but not any user, hence any other persona should not be appeared." + "Please summarize the content after this sentence and just reply with the summarization without any other content or responses." + user_input
    try:
        response = agent_executor.invoke({
            "chat_history": chat_history,
            "input": formatted_input
        })
        output = response['output']
        return output
    except:
        return None

def data_analysis(client: OpenAI, agent_executor: AgentExecutor, chat_history, user_input):
    # call acquire history data
    date_dict = get_start_end_date_for_history(client=client, user_input=user_input)
    print("--- selected date", date_dict) if __DEBUGGING__ else None
    data_csv = get_csv_given_date(csv_path=__USER_DATA_PATH__, start_date=date_dict['start_date'], end_date=date_dict['end_date'])
    print("--- data_csv\n", data_csv) if __DEBUGGING__ else None
    
    contexted_input = "The current datetime is " + datetime.datetime.now().strftime("%A, %B %d, %Y %H:%M:%S") + ". " \
                      + user_input + "User's past expense data are as follows: " + data_csv
    try:
        response = agent_executor.invoke({
            "chat_history": chat_history,
            "input": contexted_input
        })
        output = response['output']
        return output
    except:
        return None



def expenses_prediction(client: OpenAI, agent_executor: AgentExecutor, chat_history, user_input):
    # call acquire history data
    date_dict = get_start_end_date_for_history(client=client, user_input=user_input)
    print("--- selected date", date_dict) if __DEBUGGING__ else None
    data_csv = get_csv_given_date(csv_path=__USER_DATA_PATH__, start_date=date_dict['start_date'], end_date=date_dict['end_date'])
    print("--- data_csv\n", data_csv) if __DEBUGGING__ else None
    
    contexted_input = "The current datetime is " + datetime.datetime.now().strftime("%A, %B %d, %Y %H:%M:%S") + ". " \
                      + user_input + "User's past expense data are as follows: " + data_csv
    try:
        response = agent_executor.invoke({
            "chat_history": chat_history,
            "input": contexted_input
        })
        output = response['output']
        return output
    except:
        return None


# print(completion.choices[0].message.function_call.name)
# print(completion.choices[0].message.function_call.arguments)

# print(json.loads(chat_gpt(prompt)))


if __name__ == "__main__":
    # # hard codes
    # user_name = "michelle"
    # user_age = 3
    # user_mbti = "enfj-t"
    # user_gender = "female"
    # user_address = "50603 Kuala Lumpur, Wilayah Persekutuan Kuala Lumpur"
    # user_set_model_name = "Lily"
    
    # user_name = "Tom"
    # user_age = 15
    # user_mbti = "ENTP"
    # user_gender = "male"
    # user_address = "Bukit Mertajam, Pulau Pinang"
    # user_set_model_name = "Lily"
    
    user_name = "Zhi En"
    user_age = 21
    user_mbti = "INTJ"
    user_gender = "female"
    user_address = "Ipoh"
    user_set_model_name = "Lily"
    
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
            chat_history.append(AIMessage(content=formatted_output)) if formatted_output is not None else None
            # must let model know that it recorded the expense
            
        if 'expense_prediction' in funcs_to_call.keys():
            output = None
            c = 0
            while (output is None) and (c < __MAX_ERROR_TRIAL__):
                output = expenses_prediction(client=client, agent_executor=model_expenses_prediction, chat_history=chat_history, user_input=prompt)
                c += 1
            print("\n--- expenses_prediction()") if __DEBUGGING__ else None
            print(output)
            chat_history.append(AIMessage(content=output)) if output is not None else None
    
        if 'data_analysis' in funcs_to_call.keys():
            output = None
            c = 0
            while (output is None) and (c < __MAX_ERROR_TRIAL__):
                output = data_analysis(client=client, agent_executor=model_data_analysis, chat_history=chat_history, user_input=prompt)
                c += 1
            print("\n--- data_analysis()") if __DEBUGGING__ else None
            print(output)
            chat_history.append(AIMessage(content=output)) if output is not None else None
    
        # final revert
        if 'regular_chat' in funcs_to_call.keys():
            output = None
            c = 0
            while (output is None) and (c < __MAX_ERROR_TRIAL__):
                output = regular_chat(agent_executor=model_regular_chat, chat_history=chat_history, user_input=prompt)
                c += 1
            print("\n--- regular_chat()") if __DEBUGGING__ else None
            print(output)
            chat_history.append(AIMessage(content=output)) if output is not None else None

