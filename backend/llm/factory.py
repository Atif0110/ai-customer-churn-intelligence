import os
from dotenv import load_dotenv
from backend.llm.providers.groq_provider import GroqProvider

load_dotenv()


def get_llm():

    provider = os.getenv("LLM_PROVIDER", "groq")

    if provider == "groq":
        return GroqProvider()

    raise ValueError("Unsupported provider")
