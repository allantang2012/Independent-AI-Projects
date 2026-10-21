
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv()

# Tool available to our agent
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return their product."""
    print(f"[TOOL CALLED] multiply({a}, {b})")
    return a * b


# Initialize language model
llm = ChatOpenAI(
    model="gpt-5-mini",
    max_retries=2
)

# Create tool-calling agent
agent = create_agent(
    model=llm,
    tools=[multiply]
)

# Ask a question requiring tool use
result = agent.invoke({
    "messages": [
        ("user", "What is 37 multiplied by 48? Use the multiply tool.")
    ]
})

print("\n--- Agent Conversation ---")

for message in result["messages"]:
    print(f"{message.type}: {message.content}")
