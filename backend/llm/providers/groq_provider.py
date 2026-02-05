import os
from groq import Groq
from backend.llm.base import BaseLLM


class GroqProvider(BaseLLM):

    def __init__(self):
        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )

    def generate(self, prompt: str) -> str:
        chat = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        return chat.choices[0].message.content
