import asyncio

from exp import generate_toc
from preprocessing import calculate_tfidf, preprocess_text
from prompts import generate_prompt, llm
from scraper import parse_webpage

text = asyncio.run(parse_webpage("https://simple.wikipedia.org/wiki/Alexander_the_Great"))

text = preprocess_text(text)
tfidf = calculate_tfidf([text])

topic = "Mythology"
prompt = generate_prompt(topic, tfidf, 10, False)
print(prompt)
print("---")
res = asyncio.run(llm(prompt, topic))
print(res)
