from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from tools import *
load_dotenv()

llm = ChatGoogleGenerativeAI(
    api_key=os.getenv("gemini_key"),
    model="gemini-3.1-flash-lite"
)

#Binding the llm with tools
llm = llm.bind_tools([addNumbers,multiplyNumbers,saveFile,readFile])

tool_registry = {
    "addNumbers": addNumbers,
    "multiplyNumbers": multiplyNumbers,
    "saveFile": saveFile,
    "readFile": readFile
}

while True:
    user_input = input("You: ")
    if user_input.lower()=='exit':
        print("Agent : Cya!")
        break
    #Connected to llm.
    responses = llm.invoke(f'''
    -Question :{user_input}
    ''')

    if responses.tool_calls:
        tool_name = responses.tool_calls[0]['name']
        tool_args = responses.tool_calls[0]['args']
                  
        print("Agent is using tool :",tool_name)
        result = tool_registry.get(tool_name).invoke(tool_args)

        print("Agent Response :",result)
    else:
        print("Agent : Sorry I cann't help with that")