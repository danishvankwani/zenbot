from langchain_groq import ChatGroq
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage
)
from tools import calculator, knowledge_search
from dotenv import load_dotenv

load_dotenv()


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


tools = [
    calculator,
    knowledge_search
]

tool_map = {
    "calculator": calculator,
    "knowledge_search": knowledge_search
}

llm_with_tools = llm.bind_tools(tools)


SYSTEM_PROMPT = """
You are AtomBot, a helpful assistant.

You have access to exactly two tools:

1. calculator
2. knowledge_search

Use these tools when necessary.

Do not use web search or any other external tool.

After receiving a tool result, answer the user's question
using that result.

When providing the final answer, do not call any tool.
Return only the final answer in normal text.
"""


def run_agent(question):

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question)
    ]

    response = llm_with_tools.invoke(messages)

    if not response.tool_calls:
        return response

    messages.append(response)

    for tool_call in response.tool_calls:

        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        tool = tool_map[tool_name]

        tool_result = tool.invoke(tool_args)

        messages.append(
            ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"]
            )
        )

    final_response = llm.invoke(messages)

    return final_response