# created based on https://python.langchain.com/docs/get_started/quickstart

# pip install langchain
# pip install langchain-openai
# export OPENAI_API_KEY="..."
# pip install beautifulsoup4
# pip install faiss-cpu
# export TAVILY_API_KEY=...
# pip install langchainhub
# pip install "langserve[all]"
# pip install numexpr

# langsmith: to debug & inspect what is exactly going on inside the chain/agent. 
# not needed for now, but helpful for debugging llm chain if required.
# https://smith.langchain.com/
# export LANGCHAIN_TRACING_V2="true"
# export LANGCHAIN_API_KEY="..."

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor

import os
from api_keys import OPENAI_API_KEY, TAVILY_API_KEY


llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
# llm.invoke("how can langsmith help with testing?")

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are world class technical documentation writer."),
#     ("user", "{input}")
# ])

# format model output into python string for easy accessing
output_parser = StrOutputParser()

 # combine prompt and model into a LLM chain
# chain = prompt | llm | output_parser

# print(chain.invoke({"input": "how can langsmith help with testing?"}))

# web-based loader
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
docs = loader.load()

# vectorstore knowledgebase of web-loaded data
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)


prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
<context>
{context}
</context>
Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)

# document_chain.invoke({
#     "input": "how can langsmith help with testing?",
#     "context": [Document(page_content="langsmith can let you visualize test results")]
# })

retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
# print(response["answer"])


# First we need a prompt that we can pass into an LLM to generate this search query
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
])
retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

# We can test this out by passing in an instance where the user is asking a follow up question.
chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
# print(retriever_chain.invoke({
#     "chat_history": chat_history,
#     "input": "Tell me how"
# }))

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

# print(retrieval_chain.invoke({
#     "chat_history": chat_history,
#     "input": "Tell me how"
# }))


retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
)

# online search tool as knowledge source
# https://python.langchain.com/docs/integrations/retrievers/tavily
# export TAVILY_API_KEY=
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
search = TavilySearchResults()

# list of tools to work with
tools = [retriever_tool, search]

# Get the preset prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

response = agent_executor.invoke({"input": "what is the weather in SF?"})
print(response)

# chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
# agent_executor.invoke({
#     "chat_history": chat_history,
#     "input": "Tell me how"
# })