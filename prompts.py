import os
import random

from dotenv import load_dotenv
from groq import AsyncGroq

load_dotenv()

SYSTEM_PROMPT = """
You are an AI assistant specialized in domain analysis and content generation. Your primary functions are to assist with the following tasks:
Use the extracted TF-IDF weights to produce high-quality, domain-specific {} content.
- Adapt your language and approach based on the specific nature of the domain, whether it's a technical field, an academic discipline, or a more general subject area.
Do not use any markdown or latex formatting. Answer in plain text"""


def generate_prompt(topic, tfidf_scores, num=50, sorted=False):
    if not sorted:
        items = random.sample(tfidf_scores, num)
    else:
        tfidf_scores[:num]
    prompt = f"Write an unwritten text about {topic} that incorporates the following terms with their associated weights:\n"
    for term, weight in items:
        prompt += f"{term}: {weight:.4f}\n"
    prompt += f"Ensure that the generated text captures the semantic depth and accuracy of the {topic} domain."
    return prompt


async def llm(user_prompt: str, topic) -> str:
    client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(topic)},
        {"role": "user", "content": user_prompt},
    ]

    chat_completion = await client.chat.completions.create(
        messages=messages,
        model="llama3-70b-8192",
        temperature=0.3,
        max_tokens=360,
        top_p=1,
        stop=None,
        stream=False,
    )

    return chat_completion.choices[0].message.content
