import asyncio
import io
import re

import aiohttp
import pymupdf
from bs4 import BeautifulSoup, Comment, Declaration, NavigableString


def is_visible_text(element):
    if element.parent.name in {"style", "script", "head", "title", "meta", "[document]"}:
        return False
    if isinstance(element, (Comment, Declaration)):
        return False
    if isinstance(element, NavigableString):
        text = str(element).strip()
        if not text or len(text) < 25:
            return False
        if re.match(r"^\s*$", text):
            return False
    return True


def extract_visible_text(soup):
    texts = soup.findAll(text=True)
    visible_texts = filter(is_visible_text, texts)
    return " ".join(text.strip() for text in visible_texts)


async def parse_webpage(url: str):
    if url.endswith(".pdf"):
        return await parse_pdf_from_url(url)
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            content = await response.read()
            response.raise_for_status()
            # soup = bs(content, "html.parser").findAll(string=True)
            soup = BeautifulSoup(content, "html.parser")
            text = extract_visible_text(soup)
            return text


def parse_pdf(pdf, buffer=False):
    if buffer:
        doc = pymupdf.open(stream=pdf, filetype="pdf")
    else:
        doc = pymupdf.open(pdf)

    text = ""
    for page in doc:
        text += page.get_text()
    return text


async def parse_pdf_from_url(url, max_retries=3):
    async with aiohttp.ClientSession() as session:
        for attempt in range(max_retries):
            try:
                headers = {"Accept": "application/pdf"}
                async with session.get(url, headers=headers, timeout=10) as r:
                    r.raise_for_status()
                    content = await r.read()
                    pdf = io.BytesIO(content)
                    return parse_pdf(pdf, True)
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** (attempt + 1))
                else:
                    print(f"Max retries reached. Unable to download PDF from {url}")
        return None
