import re
import asyncio
import concurrent.futures
import numpy as np
from typing import List
from functools import lru_cache
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from utils import get_pdf_text_from_url
from rank_bm25 import BM25Okapi

class HybridRetriever:
    def __init__(self, documents, embedding_model, alpha=0.65):
        self.docs = documents
        self.alpha = alpha
        
        doc_texts = [doc.page_content for doc in documents]
        tokenized_docs = [text.lower().split() for text in doc_texts]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        self.vector_store = FAISS.from_documents(documents, embedding_model)
        self.semantic_retriever = self.vector_store.as_retriever(search_kwargs={"k": 8})
        
    def retrieve(self, query: str, k: int = 4):
        query_lower = query.lower()
        tokenized_query = query_lower.split()
        
        bm25_scores = self.bm25.get_scores(tokenized_query)
        semantic_docs = self.vector_store.similarity_search_with_score(query, k=k*2)
        
        final_scores = {}
        
        if len(bm25_scores) > 0:
            bm25_max = max(bm25_scores)
            if bm25_max > 0:
                for i, score in enumerate(bm25_scores):
                    final_scores[i] = (1 - self.alpha) * (score / bm25_max)
        
        for doc, distance in semantic_docs:
            try:
                doc_idx = next(i for i, d in enumerate(self.docs) if d.page_content == doc.page_content)
                semantic_score = 1 / (1 + distance)
                if doc_idx in final_scores:
                    final_scores[doc_idx] += self.alpha * semantic_score
                else:
                    final_scores[doc_idx] = self.alpha * semantic_score
            except:
                continue
        
        top_indices = sorted(final_scores.keys(), key=lambda x: final_scores[x], reverse=True)[:k]
        return [self.docs[i] for i in top_indices]

def smart_chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 80) -> List[str]:
    try:
        patterns = [
            r'^\s*\d+\.\s',
            r'^\s*[a-zA-Z]\.\s',
            r'^\s*•\s',
            r'^\s*Section\s\w+',
            r'^\s*Clause\s\w+',
            r'\n\s*\n'
        ]
        
        splits = []
        current_pos = 0
        
        for pattern in patterns:
            matches = list(re.finditer(pattern, text, re.MULTILINE))
            for match in matches:
                if match.start() > current_pos:
                    splits.append((current_pos, match.start()))
                current_pos = match.start()
        
        if current_pos < len(text):
            splits.append((current_pos, len(text)))
        
        chunks = []
        for start, end in splits:
            chunk = text[start:end].strip()
            if len(chunk) > 50:
                chunks.append(chunk)
        
        if not chunks:
            chunks = [text]
        
    except Exception as e:
        print(f"Pattern splitting failed: {e}")
        chunks = [text]

    final_chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ";", ",", " "]
    )
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for chunk in chunks:
            if len(chunk) > chunk_size:
                future = executor.submit(splitter.split_text, chunk)
                futures.append(future)
            else:
                final_chunks.append(chunk)
        
        for future in concurrent.futures.as_completed(futures):
            try:
                final_chunks.extend(future.result())
            except:
                pass
    
    final_chunks = [c for c in final_chunks if len(c.strip()) > 30]
    return final_chunks if final_chunks else [text[:chunk_size]]

class OptimizedRAGCore:
    def __init__(self, document_url: str, embedding_model, llm):
        print(f"🔄 Setting up optimized RAG for: {document_url}")
        self.llm = llm
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        try:
            raw_text = get_pdf_text_from_url(document_url)
            if not raw_text:
                raise ValueError("PDF read failed")
        except Exception as e:
            print(f"PDF processing error: {e}")
            raise ValueError(f"Document processing failed: {str(e)}")

        try:
            print("📝 Smart chunking...")
            chunks = smart_chunk_text(raw_text, chunk_size=800, chunk_overlap=80)
            documents = [Document(page_content=chunk) for chunk in chunks]
            print(f"✅ Created {len(documents)} optimized chunks")
        except Exception as e:
            print(f"Chunking error: {e}")
            documents = [Document(page_content=raw_text[:800])]

        try:
            print("🧠 Building hybrid index...")
            self.retriever = HybridRetriever(documents, embedding_model, alpha=0.65)
            print("✅ Hybrid retriever ready")
        except Exception as e:
            print(f"Retriever error: {e}")
            self.vector_store = FAISS.from_documents(documents, embedding_model)
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})

        prompt_template = """Based on the insurance document context, provide a precise answer.

Context: {context}

Question: {input}

Answer:"""

        prompt = ChatPromptTemplate.from_template(prompt_template)
        self.qa_chain = create_stuff_documents_chain(self.llm, prompt)
        print("✅ QA system ready")

    def _safe_retrieve(self, question: str):
        try:
            if hasattr(self.retriever, 'retrieve'):
                return self.retriever.retrieve(question, k=4)
            else:
                return self.retriever.invoke(question)
        except Exception as e:
            print(f"Retrieval error: {e}")
            return []

    def answer_question(self, question: str) -> str:
        try:
            docs = self._safe_retrieve(question)
            if not docs:
                return "Unable to find relevant information in the document."
            
            response = self.qa_chain.invoke({
                "input": question,
                "context": docs
            })
            
            return response.strip()
        except Exception as e:
            print(f"Answer generation error: {e}")
            return f"Error processing question: {str(e)}"

    async def answer_questions_batch(self, questions: List[str]) -> List[str]:
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(self.executor, self.answer_question, q) 
            for q in questions
        ]
        return await asyncio.gather(*tasks)