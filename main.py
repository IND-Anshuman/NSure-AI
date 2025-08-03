import warnings
import os
import asyncio
import hashlib
import time
import functools
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Depends, HTTPException, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Dict

warnings.filterwarnings("ignore")

os.environ["HF_HOME"] = "/tmp/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache/transformers"

model_cache = {}
pipeline_cache = {}
executor = ThreadPoolExecutor(max_workers=4)

def timed_cache(maxsize=32, ttl=1800):
    def decorator(func):
        cache = {}
        cache_times = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(sorted(kwargs.items()))
            current_time = time.time()
            
            if key in cache and current_time - cache_times[key] < ttl:
                return cache[key]
            
            result = func(*args, **kwargs)
            cache[key] = result
            cache_times[key] = current_time
            
            if len(cache) > maxsize:
                oldest_key = min(cache_times.keys(), key=lambda k: cache_times[k])
                del cache[oldest_key]
                del cache_times[oldest_key]
            
            return result
        return wrapper
    return decorator

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- Loading models ---")
    from sentence_transformers import SentenceTransformer
    from langchain_google_genai import ChatGoogleGenerativeAI
    from dotenv import load_dotenv
    
    load_dotenv()
    
    try:
        model_cache["embedding_model"] = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("✅ Embeddings loaded")

        model_cache["llm"] = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro", 
            temperature=0.0,
            max_tokens=200,
            timeout=30,
            max_retries=3,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        print("✅ LLM loaded")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise e

    yield
    
    executor.shutdown(wait=True)
    model_cache.clear()
    pipeline_cache.clear()

app = FastAPI(
    title="NSure-AI: Smart Insurance Assistant",
    description="Upload insurance PDFs and get instant answers",
    version="1.0.0",
    lifespan=lifespan
)

from rag_core import OptimizedRAGCore

API_TOKEN = "ee3aca9314e8c88b242c5f86bdb52d0bbb80293d95ced9beb6553a7fbb8cd1ce"
bearer_auth = HTTPBearer()

def check_auth(credentials: HTTPAuthorizationCredentials = Security(bearer_auth)):
    if credentials.scheme != "Bearer" or credentials.credentials != API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    return credentials.credentials

class QueryRequest(BaseModel):
    documents: str = Field(..., description="URL to PDF document")
    questions: List[str] = Field(..., min_length=1, description="Questions to answer")

class QueryResponse(BaseModel):
    answers: List[str]

@app.get("/", include_in_schema=False)
def home():
    return {"status": "NSure-AI is running"}

@app.post("/hackrx/run", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    token: str = Depends(check_auth)
):
    try:
        rag = OptimizedRAGCore(
            embedding_model=model_cache["embedding_model"],
            llm=model_cache["llm"]
        )
        
        answers = await rag.process_queries(request.documents, request.questions)
        return QueryResponse(answers=answers)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))