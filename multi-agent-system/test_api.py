
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load API key securely from .env
load_dotenv()

# Initialize the language model
llm = ChatOpenAI(
    model="gpt-5-mini",
    max_retries=2
)

# Send a test request
print("Connecting to OpenAI...")

response = llm.invoke(
    "Explain what an AI agent is in one sentence."
)

print("\n--- OpenAI API Test ---")
print(response.content)

# Track token usage
print("\n--- Token Usage ---")
print(response.usage_metadata)
