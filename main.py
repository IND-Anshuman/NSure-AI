# main.py
import warnings
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict

# Hide annoying warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Cache settings for HuggingFace models
os.environ["HF_HOME"] = "/tmp/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface_cache/transformers"
os.environ["HF_HUB_CACHE"] = "/tmp/huggingface_cache/hub"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/tmp/huggingface_cache/sentence_transformers"

# Store models and pipelines in memory
model_cache = {}
pipeline_cache: Dict[str, "RAGCore"] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models when server starts
    print("--- Loading AI models... ---")
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
    from dotenv import load_dotenv
    import tempfile

    load_dotenv()

    # Use temp dir for model storage
    temp_cache_dir = tempfile.mkdtemp(prefix="hf_cache_")
    print(f"Using cache: {temp_cache_dir}")

    try:
        # Load embedding model
        model_cache["embedding_model"] = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            cache_folder=temp_cache_dir
        )
        print("✅ Embeddings loaded")

        model_cache["llm"] = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        print("✅ LLM loaded")
        print("🚀 Ready to serve requests!")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        # Fallback without custom cache
        try:
            model_cache["embedding_model"] = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            print("✅ Embeddings loaded (fallback)")
        except Exception as e2:
            print(f"❌ Complete failure: {e2}")
            raise e2

    yield

    # Cleanup when server shuts down
    print("🔄 Shutting down...")
    model_cache.clear()
    pipeline_cache.clear()

# Create FastAPI app
app = FastAPI(
    title="NSure-AI: Smart Insurance Assistant",
    description="Upload insurance PDFs and get instant answers to your questions",
    version="1.0.0",
    lifespan=lifespan
)

# Import our RAG system
from rag_core import RAGCore
from fastapi import Depends, HTTPException, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Authentication setup
API_TOKEN = "ee3aca9314e8c88b242c5f86bdb52d0bbb80293d95ced9beb6553a7fbb8cd1ce"
bearer_auth = HTTPBearer()

def check_auth(credentials: HTTPAuthorizationCredentials = Security(bearer_auth)):
    if credentials.scheme != "Bearer" or credentials.credentials != API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    return credentials.credentials

# Request/Response models
class QueryRequest(BaseModel):
    documents: str = Field(..., description="URL to PDF document")
    questions: List[str] = Field(..., min_length=1, description="Questions to answer")

class QueryResponse(BaseModel):
    answers: List[str]

# API endpoints
@app.get("/", include_in_schema=False)
def home():
    return {"message": "NSure-AI is running! Check /docs for API info."}

@app.post("/hackrx/run", response_model=QueryResponse, tags=["Main"])
async def process_document(request: QueryRequest, token: str = Depends(check_auth)):
    doc_url = request.documents

    # Check if we already processed this document
    if doc_url in pipeline_cache:
        rag = pipeline_cache[doc_url]
        print(f"📋 Using cached pipeline for: {doc_url}")
    else:
        print(f"🔄 Creating new pipeline for: {doc_url}")
        try:
            rag = RAGCore(
                document_url=doc_url,
                embedding_model=model_cache["embedding_model"],
                llm=model_cache["llm"]
            )
            pipeline_cache[doc_url] = rag
        except Exception as e:
            print(f"❌ Pipeline creation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

    # Get answers for all questions
    answers = []
    for q in request.questions:
        answer = rag.answer_question(q)
        answers.append(answer)

    return QueryResponse(answers=answers)