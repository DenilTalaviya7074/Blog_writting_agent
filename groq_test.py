import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

# Get API Key
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

# Initialize LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=api_key
)

try:
    response = llm.invoke(
        [
            HumanMessage(
                content="Say 'Groq API is working!' and tell me today's capital of France."
            )
        ]
    )

    print("=" * 50)
    print("✅ API Connected Successfully")
    print("=" * 50)
    print(response.content)

except Exception as e:
    print("❌ Error")
    print(type(e).__name__)
    print(e)