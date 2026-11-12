import os
from dotenv import load_dotenv

from typesafe_sdk import TypeSafeClient, Choice
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage

from tools import *

load_dotenv()

# Gemini setup
llm = ChatGoogleGenerativeAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    model="gemini-3.1-flash-lite"
)

# Available LangChain tools
TOOLS = [addNumbers,multiplyNumbers,saveFile,readFile]

# Tool descriptions for Jev
TOOL_CRITERIA = {
    "addNumbers": "Add two or more numbers.",
    "multiplyNumbers": "Multiply two or more numbers.",
    "saveFile": "Save or write content into a file.",
    "readFile": "Read or retrieve content from a file."
}

# Tool lookup
tool_map = {tool.name: tool for tool in TOOLS}

# Jev tool selection
def select_tool(client, user_input):
    response = client.system_one(
        state={ #it's important
            "user_request": user_input
        },
        questions={
            "tool_selection": Choice(
                instructions=(
                    "Select the single most appropriate tool "
                    "for the user's request. Select general if "
                    "none of the available tools is required."
                ),
                criteria=TOOL_CRITERIA
            )
        }
    )

    decision = response.choices["tool_selection"]

    print("\n--- Jev Decision ---")
    print("Selected Tool:", decision.choice)
    print("Confidence:", decision.confidence)
    print("Probabilities:", decision.probabilities)

    return decision.choice, decision.confidence


# Gemini generates arguments and Python executes the tool
def execute_tool(tool_name, user_input):
    selected_tool = tool_map[tool_name]
    # Bind only the tool selected by Jev
    agent = llm.bind_tools([selected_tool])
    # Ask Gemini to generate tool arguments
    response = agent.invoke([
        HumanMessage(content=user_input)
    ])

    if not response.tool_calls:
        print("\nAgent:", response.content)
        return

    # Execute selected tool
    for tool_call in response.tool_calls:

        if tool_call["name"] != tool_name:
            continue

        result = selected_tool.invoke(tool_call["args"])

        print("\n--- Tool Execution ---")
        print("Tool:", tool_name)
        print("Arguments:", tool_call["args"])
        print("Result:", result)

        # Send tool result back to Gemini
        # final_response = llm.invoke([
        #     HumanMessage(content=user_input),
        #     response,
        #     ToolMessage(
        #         content=str(result),
        #         tool_call_id=tool_call["id"]
        #     )
        # ])

        # print("\nAgent:", final_response.content[0]["text"])


# Main chat loop
def main():
    print("Jev + Gemini Agent Started!")
    print("Type 'exit' to quit.")

    with TypeSafeClient() as client:
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() == "exit":
                print("Agent: Cya!")
                break
            if not user_input:
                continue
            try:

                # Step 1: Jev selects a tool
                selected_tool, confidence = select_tool(
                    client,
                    user_input
                )

                # Step 2: Handle general conversation
                if selected_tool == "general":
                    response = llm.invoke(user_input)
                    print("\nAgent:", response.content)
                    continue

                # Step 3: Validate Jev's selection
                if selected_tool not in tool_map:
                    print("Agent: Invalid tool selected by Jev.")
                    continue

                if confidence < 0.55:
                    print("Agent: Jev is uncertain about tool selection.")
                    continue

                # Step 4: Gemini generates arguments
                # Step 5: Python executes the tool
                # Step 6: Gemini generates final response
                execute_tool(selected_tool, user_input)

            except Exception as e:
                print("\nAgent Error:", e)

if __name__ == "__main__":
    main()
