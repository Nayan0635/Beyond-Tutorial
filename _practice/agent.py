from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.chat_models import init_chat_model

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 

from pydantic import BaseModel, Field #for structured output

# from langchain_core.tools import tool
from tools import *

from dotenv import load_dotenv
import os

load_dotenv()

model = init_chat_model(
    "gemini-3.6-flash-lite",
    api_key = os.getenv("GEMINI_API_KEY")
)

# >>tools.py


tool_list = [currency_converter, weight_converter, budget_calculator]

prompt = ChatPromptTemplate.from_messages([
    ("system", '''
        You are a shopping assistant that converts units, converts currency, calculates tax, and checks budgets.
        If a tool returns an error, explain it politely to the user and ask for the correct unit/currency.
    '''),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

class shopping(BaseModel):
    Price: float = Field(description=" Price of the items that the customer want to buy")
    Tax: float = Field(description=" Tax amount on the item")
    Total_cost: float = Field(description=" total Cost of the items including tax")
    Budget: float = Field(description="The customer's budget")
    steps: list = Field(description="Raw tool observations")

structured_model = model.with_structured_output(shopping)

agent = create_tool_calling_agent(model, tool_list, prompt)
agent_executor = AgentExecutor(agent= agent, tools= tool_list, return_intermediate_steps= True, verbose=True,)

# AgentExecutor keeps a record of intermediate_steps
def _steps(intermediate_steps) -> list[str]:
    ''' Return the observation of the steps, exactly as the stool produce it'''
    return [observation for _action, observation in intermediate_steps]

while True:
    user_input = input("Ask? ")
    if user_input.strip().lower() == "exit":
        break

    response = agent_executor.invoke({"input": user_input})
    steps = _steps(response.get("intermediate_steps", []))

    result = structured_model.invoke(
        f"""
        Convert the following shopping assistant result into the
        required structured output schema.

        User request: {user_input}

        Agent's final answer: {response["output"]}

        Tool observations: {steps}

        Extract the available price, tax, total cost, and budget.
        Put the tool observations into the steps field.
        Do not invent missing values.
        """
    )

    print(result)
