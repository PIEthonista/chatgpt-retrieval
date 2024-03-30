# conda env setup

- conda create -n exchangedaonocolor python=3.10
- conda activate exchangedaonocolor
- conda install langchain
- pip install langchain-openai
- pip install langchain_openai
- pip install beautifulsoup4
- pip install faiss-cpu
- pip install langchainhub
- pip install "langserve[all]"
- pip install numexpr
- pip install flask
- pip install flask_cors
- pip install openai
- pip install pandas
- pip install tavily-python

In order to run 'python server.py'
Please ensure you type 'ipconfig' in terminal - get the ipv4 addess
Replace the ip address in "CORS(app, resources={r"/_": {"origins": ["http://10.168.105.128:5000", "_"]}})" with the ipv4 address
You may change the port if you want, but have to ensure the ipv4 address with the port is the same in 'sendMessageToChatGPT.js'
However, do not change the get_user_response after the ipv4 address

Please take note that you have to create a file named 'api_keys.py' with 2 keys
OPENAI_API_KEY = "Your key"
TAVILY_API_KEY = "Your key"

</br>

# API Key file

- please place "api_keys.py" at the same level of directory as this README.md file.

</br>

# To Serve Backend

- ?
