# main.py
import warnings
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict

# Suppress known warnings to keep logs clean
warnings.filterwarnings("ignore", category=FutureWarning)

# Set cache directory environment variables
os.environ["HF_HOME"] = "/tmp/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface_cache/transformers"
os.environ["HF_HUB_CACHE"] = "/tmp/huggingface_cache/hub"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/tmp/huggingface_cache/sentence_transformers"

# --- Global State & Model Cache ---
model_cache = {}
pipeline_cache: Dict[str, "RAGCore"] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    print("--- Application Startup: Downloading and loading models... ---")
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
    from dotenv import load_dotenv
    import tempfile

    load_dotenv()

    # Create a temporary directory for model cache
    temp_cache_dir = tempfile.mkdtemp(prefix="hf_cache_")
    print(f"Using temporary cache directory: {temp_cache_dir}")

    try:
        # Initialize embedding model with explicit cache directory
        model_cache["embedding_model"] = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            cache_folder=temp_cache_dir
        )
        print("   -> Embedding model loaded successfully.")

        model_cache["llm"] = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        print("   -> LLM loaded.")
        print("--- Model loading complete. Application is ready. ---")
    except Exception as e:
        print(f"Error loading models: {e}")
        # Try alternative approach without cache_folder
        try:
            model_cache["embedding_model"] = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            print("   -> Embedding model loaded with default cache.")
        except Exception as e2:
            print(f"Failed to load embedding model: {e2}")
            raise e2

    yield

    # --- Shutdown ---
    print("--- Application Shutdown ---")
    model_cache.clear()
    pipeline_cache.clear()

# --- Application Setup ---
app = FastAPI(
    title="NSure-AI: Intelligent Query-Retrieval System",
    lifespan=lifespan
)

# --- Dynamic Import for RAGCore ---
from rag_core import RAGCore
from fastapi import Depends, HTTPException, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# --- Security & Authentication ---
REQUIRED_BEARER_TOKEN = "ee3aca9314e8c88b242c5f86bdb52d0bbb80293d95ced9beb6553a7fbb8cd1ce"
bearer_scheme = HTTPBearer()

def validate_token(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)):
    if credentials.scheme != "Bearer" or credentials.credentials != REQUIRED_BEARER_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication token"
        )
    return credentials.credentials

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    documents: str = Field(..., description="A public URL to the PDF document.")
    questions: List[str] = Field(..., min_length=1, description="List of questions.")

class QueryResponse(BaseModel):
    answers: List[str]

# --- API Endpoints ---
@app.get("/", include_in_schema=False)
def read_root():
    return {"status": "NSure-AI API is ready."}

@app.post("/hackrx/run", response_model=QueryResponse, tags=["Core Functionality"])
async def run_submission(request: QueryRequest, token: str = Depends(validate_token)):
    doc_url = request.documents

    if doc_url in pipeline_cache:
        rag_pipeline = pipeline_cache[doc_url]
        print(f"Cache HIT: Using existing RAG pipeline for URL: {doc_url}")
    else:
        print(f"Cache MISS: Initializing new RAG pipeline for URL: {doc_url}")
        try:
            rag_pipeline = RAGCore(
                document_url=doc_url,
                embedding_model=model_cache["embedding_model"],
                llm=model_cache["llm"]
            )
            pipeline_cache[doc_url] = rag_pipeline
        except Exception as e:
            print(f"Error during RAGCore initialization: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

    answers = []
    for question in request.questions:
        answer = rag_pipeline.answer_question(question)
        answers.append(answer)

    return QueryResponse(answers=answers)