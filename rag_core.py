import re
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from utils import get_pdf_text_from_url

def smart_chunk_text(text: str, chunk_size: int, chunk_overlap: int):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )
    
    chunks = splitter.split_text(text)
    print(f"Split into {len(chunks)} chunks")
    return chunks

class RAGCore:
    def __init__(self, document_url: str, embedding_model, llm):
        print(f"🔄 Setting up RAG for: {document_url}")
        self.llm = llm

        raw_text = get_pdf_text_from_url(document_url)
        if not raw_text:
            raise ValueError("Couldn't read PDF - check the URL")

        print("📝 Chunking document...")
        chunks = smart_chunk_text(raw_text, chunk_size=1000, chunk_overlap=200)
        documents = [Document(page_content=chunk) for chunk in chunks]
        print(f"✅ Created {len(documents)} chunks")

        print("🧠 Building vector index...")
        self.vector_store = FAISS.from_documents(documents, embedding_model)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        print("✅ Vector index ready")

        prompt_template = """You're an expert at reading insurance documents. Answer the question based ONLY on the provided context.

Rules:
- Be specific and accurate
- Quote exact text when possible
- If the answer isn't in the context, say "Information not found in the document"
- Don't make assumptions or add external knowledge

Context from document:
{context}

Question: {input}

Answer:"""

        prompt = ChatPromptTemplate.from_template(prompt_template)
        self.qa_chain = create_stuff_documents_chain(self.llm, prompt)
        print("✅ QA system ready")

    def answer_question(self, question: str) -> str:
        print(f"❓ Question: {question}")
        
        docs = self.retriever.invoke(question)
        
        response = self.qa_chain.invoke({
            "input": question,
            "context": docs
        })
        
        return response.strip()