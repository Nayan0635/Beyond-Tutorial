"""Compare Gemini tool selection (baseline) with Jev-routed tool selection."""

import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage
from typesafe_sdk import Choice, TypeSafeClient

from tools import addNumbers, multiplyNumbers, saveFile, readFile


load_dotenv()

MODEL_NAME = "gemini-3.1-flash-lite"
CONFIDENCE_THRESHOLD = 0.70 

TOOLS = {
    "addNumbers": addNumbers,
    "multiplyNumbers": multiplyNumbers,
    "saveFile": saveFile,
    "readFile": readFile,
}

# Jev chooses one of the four tools, or explicitly says no tool is suitable.
ROUTING_CHOICES = {
    "addNumbers": "Add two numbers.",
    "multiplyNumbers": "Multiply two numbers.",
    "saveFile": "Save user-provided text into hello.txt.",
    "readFile": "Read the contents of hello.txt.",
    "no_tool": "The request does not match any available tool.",
}

SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the supplied tool when appropriate. "
    "After receiving a tool result, explain the result clearly. "
    "If a tool returns an error, explain it politely."
)

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0,
)


def token_counts(message):
    """Read LangChain's per-call usage metadata."""
    usage = getattr(message, "usage_metadata", None) or {}
    return {
        "input": usage.get("input_tokens", 0),
        "output": usage.get("output_tokens", 0),
        "total": usage.get("total_tokens", 0),
    }


def format_tokens(counts):
    return f'{counts["input"]}/{counts["output"]}/{counts["total"]}'


def run_selected_tool(ai_message):
    """Execute the first tool call and return its result as a ToolMessage."""
    if not ai_message.tool_calls:
        return None, None, "No tool call was produced."

    call = ai_message.tool_calls[0]
    tool_name = call["name"]
    tool_args = call["args"]

    try:
        result = TOOLS[tool_name].invoke(tool_args)
        result_text = str(result)
    except Exception as exc:
        # tools.py is intentionally unchanged. In particular, readFile()
        # raises FileNotFoundError if hello.txt does not exist yet.
        result_text = f"Tool error: {type(exc).__name__}: {exc}"

    tool_message = ToolMessage(
        content=result_text,
        tool_call_id=call["id"],
    )
    return tool_name, tool_message, result_text


def finish_with_tool_result(model_with_tools, question, ai_message, tool_message):
    """Send the tool result back to Gemini so it can produce a final answer."""
    final_message = model_with_tools.invoke([
        ("system", SYSTEM_PROMPT),
        HumanMessage(content=question),
        ai_message,
        tool_message,
    ])
    return final_message.content


def baseline_path(question):
    """Baseline: Gemini sees all four tool schemas and selects one."""
    model_with_all_tools = llm.bind_tools(list(TOOLS.values()))
    selection_message = model_with_all_tools.invoke([
        ("system", SYSTEM_PROMPT),
        ("human", question),
    ])

    selection_usage = token_counts(selection_message)

    if not selection_message.tool_calls:
        return {
            "path": "Baseline",
            "selection": selection_usage,
            "answer": selection_message.content or "No response.",
            "tool": None,
        }

    tool_name, tool_message, _ = run_selected_tool(selection_message)
    answer = finish_with_tool_result(
        model_with_all_tools, question, selection_message, tool_message
    )

    return {
        "path": "Baseline",
        "selection": selection_usage,
        "answer": answer,
        "tool": tool_name,
    }


def jev_select_tool(question):
    """Jev performs the bounded classification; no Gemini call is made here."""
    with TypeSafeClient() as client:
        response = client.system_one(
            state=question,
            questions={
                "tool": Choice(
                    instructions=(
                        "Which single tool, if any, should handle this request? "
                        "Choose no_tool when the request is unrelated, unclear, "
                        "or lacks the information needed to call a tool."
                    ),
                    criteria=ROUTING_CHOICES,
                )
            },
        )

    answer = response.choices["tool"]
    return answer.choice, answer.confidence, response


def jev_routed_path(question):
    """Jev routes; Gemini sees only the selected tool for argument generation."""
    selected_tool, confidence, jev_response = jev_select_tool(question)

    if confidence < CONFIDENCE_THRESHOLD or selected_tool == "no_tool":
        result = baseline_path(question)
        result["path"] = (
            f"Jev → Baseline fallback ({confidence:.2f})"
        )
        result["jev_confidence"] = confidence
        result["jev_usage"] = getattr(jev_response, "usage", None)
        return result

    one_tool_model = llm.bind_tools([TOOLS[selected_tool]])
    selection_message = one_tool_model.invoke([
        ("system", SYSTEM_PROMPT),
        ("human", question),
    ])
    selection_usage = token_counts(selection_message)

    # A classifier can be confidently wrong. If Gemini does not call the
    # selected tool, use the baseline rather than silently stalling.
    if not selection_message.tool_calls:
        result = baseline_path(question)
        result["path"] = f"Jev → Baseline fallback (no call, {confidence:.2f})"
        result["jev_confidence"] = confidence
        result["jev_usage"] = getattr(jev_response, "usage", None)
        return result

    actual_tool = selection_message.tool_calls[0]["name"]
    if actual_tool != selected_tool:
        result = baseline_path(question)
        result["path"] = f"Jev → Baseline fallback (tool mismatch, {confidence:.2f})"
        result["jev_confidence"] = confidence
        result["jev_usage"] = getattr(jev_response, "usage", None)
        return result

    tool_name, tool_message, _ = run_selected_tool(selection_message)
    answer = finish_with_tool_result(
        one_tool_model, question, selection_message, tool_message
    )

    return {
        "path": f"Jev-routed ({confidence:.2f})",
        "selection": selection_usage,
        "answer": answer,
        "tool": tool_name,
        "jev_confidence": confidence,
        "jev_usage": getattr(jev_response, "usage", None),
    }


def print_comparison(rows):
    headers = ("Question", "Path", "Gemini first-call tokens I/O/Total", "Final answer")
    print("\n" + " | ".join(headers))
    print("-" * 115)

    for question, result in rows:
        print(
            f'{question} | {result["path"]} | '
            f'{format_tokens(result["selection"])} | '
            f'{str(result["answer"]).replace(chr(10), " ")[:100]}'
        )

        if "jev_usage" in result:
            # Jev usage is separate from Gemini's LangChain usage_metadata.
            print("  Jev confidence:", result.get("jev_confidence"))
            print("  Jev API usage:", result["jev_usage"])


def main():
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("gemini_key")):
        raise RuntimeError("Set GEMINI_API_KEY (or your existing gemini_key) in .env.")
    if not os.getenv("TYPESAFE_API_KEY"):
        raise RuntimeError("TYPESAFE_API_KEY is missing from your .env file.")

    questions = [
        "Read hello.txt",
        "Add 5 and 7.",
        "Multiply 6 and 9.",
        "Save this exact text in hello.txt: Jev routing test file.",
        "Read hello.txt.",
        "What is the capital of France?",
    ]

    rows = []
    for question in questions:
        print(f"\nQUESTION: {question}")

        baseline = baseline_path(question)
        routed = jev_routed_path(question)

        rows.append((question, baseline))
        rows.append((question, routed))

        print("Baseline:", baseline["answer"])
        print("Jev path:", routed["answer"])

    print_comparison(rows)


if __name__ == "__main__":
    main()
