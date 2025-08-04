import requests
import fitz
import re
from functools import lru_cache
from typing import Optional
import asyncio
import aiohttp

@lru_cache(maxsize=32)
def get_pdf_text_from_url(url: str) -> Optional[str]:
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        
        doc = fitz.open(stream=response.content, filetype="pdf")
        
        text_parts = []
        for page in doc:
            page_text = page.get_text()
            cleaned_text = re.sub(r'\s+', ' ', page_text.strip())
            if cleaned_text:
                text_parts.append(cleaned_text)
        
        doc.close()
        
        full_text = ' '.join(text_parts)
        return full_text if len(full_text) > 50 else None
        
    except Exception as e:
        print(f"PDF extraction error: {e}")
        return None

async def get_pdf_text_async(url: str) -> Optional[str]:
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
            async with session.get(url) as response:
                content = await response.read()
                
        doc = fitz.open(stream=content, filetype="pdf")
        text_parts = []
        
        for page in doc:
            page_text = page.get_text()
            cleaned_text = re.sub(r'\s+', ' ', page_text.strip())
            if cleaned_text:
                text_parts.append(cleaned_text)
        
        doc.close()
        return ' '.join(text_parts) if text_parts else None
        
    except Exception as e:
        print(f"Async PDF error: {e}")
        return None