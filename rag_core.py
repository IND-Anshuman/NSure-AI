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

def context_aware_chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> List[str]:
    """Smart chunking that preserves medical/insurance context"""
    
    # Insurance document section patterns
    section_patterns = [
        r'(?i)(coverage|benefit|exclusion|limitation|condition|procedure|treatment)',
        r'(?i)(section|clause|article|paragraph)\s+\d+',
        r'(?i)(what.*covered|what.*not.*covered|how.*works)',
        r'\d+\.\s+[A-Z]',  # Numbered sections
        r'[A-Z][a-z]+:\s*',  # Title sections
    ]
    
    # Split by sections first
    sections = []
    current_section = ""
    
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        is_new_section = any(re.search(pattern, line) for pattern in section_patterns)
        
        if is_new_section and current_section and len(current_section) > 100:
            sections.append(current_section.strip())
            current_section = line
        else:
            current_section += ' ' + line
    
    if current_section:
        sections.append(current_section.strip())
    
    # Now chunk each section preserving context
    final_chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]
    )
    
    for section in sections:
        if len(section) > chunk_size:
            sub_chunks = splitter.split_text(section)
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(section)
    
    return [c for c in final_chunks if len(c.strip()) > 50]

class HybridRetriever:
    def __init__(self, documents, embedding_model, alpha=0.65):
        self.docs = documents
        self.alpha = alpha
        self.embedding_model = embedding_model
        
        doc_texts = [doc.page_content for doc in documents]
        tokenized_docs = [text.lower().split() for text in doc_texts]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        embeddings = self.embedding_model.encode([d.page_content for d in documents])
        import faiss
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
    def retrieve(self, query: str, k: int = 4):
        query_lower = query.lower()
        tokenized_query = query_lower.split()
        
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        query_embedding = self.embedding_model.encode([query])
        import faiss
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding.astype('float32'), k*2)
        
        final_scores = {}
        
        if len(bm25_scores) > 0:
            bm25_max = max(bm25_scores)
            if bm25_max > 0:
                for i, score in enumerate(bm25_scores):
                    final_scores[i] = (1 - self.alpha) * (score / bm25_max)
        
        for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < len(self.docs):
                if idx in final_scores:
                    final_scores[idx] += self.alpha * score
                else:
                    final_scores[idx] = self.alpha * score
        
        top_indices = sorted(final_scores.keys(), key=lambda x: final_scores[x], reverse=True)[:k]
        return [self.docs[i] for i in top_indices]

def smart_chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 80) -> List[str]:
    try:
        parts = re.split(r'\n\s*\n|\d+\.\s|\w+\.\s', text)
        chunks = [p.strip() for p in parts if len(p.strip()) > 50]
        if not chunks:
            chunks = [text]
    except:
        chunks = [text]

    final_chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    
    for chunk in chunks:
        if len(chunk) > chunk_size:
            try:
                sub_chunks = splitter.split_text(chunk)
                final_chunks.extend(sub_chunks)
            except:
                final_chunks.append(chunk[:chunk_size])
        else:
            final_chunks.append(chunk)
    
    final_chunks = [c for c in final_chunks if len(c.strip()) > 30]
    return final_chunks if final_chunks else [text[:chunk_size]]

class OptimizedRAGCore:
    def __init__(self, document_url: str, embedding_model, llm):
        print(f"🔄 Setting up context-aware RAG for: {document_url}")
        self.llm = llm

        try:
            raw_text = get_pdf_text_from_url(document_url)
            if not raw_text:
                raise ValueError("PDF read failed")
        except Exception as e:
            raise ValueError(f"Document processing failed: {str(e)}")

        try:
            print("📝 Context-aware chunking...")
            chunks = context_aware_chunk_text(raw_text, chunk_size=1000, chunk_overlap=150)
            documents = [Document(page_content=chunk) for chunk in chunks]
            print(f"✅ Created {len(documents)} context-aware chunks")
        except Exception as e:
            documents = [Document(page_content=raw_text[:1000])]

        try:
            print("🧠 Building enhanced hybrid index...")
            self.retriever = EnhancedHybridRetriever(documents, embedding_model, alpha=0.6)
            print("✅ Enhanced retriever ready")
        except Exception as e:
            print(f"Enhanced retriever failed, using fallback: {e}")
            self.retriever = HybridRetriever(documents, embedding_model, alpha=0.6)

        prompt_template = """You are an expert insurance policy analyst. Analyze the provided context carefully and answer the question with high precision.

    INSTRUCTIONS:
    1. Extract ALL relevant information from the context, even if terminology differs from the question
    2. Look for synonyms, medical terms, indirect references, and related concepts
    3. If the exact term isn't found, search for related medical/insurance terminology
    4. Provide specific details including coverage amounts, conditions, exclusions, and procedures
    5. If information is partial, state what IS covered and note any limitations
    6. Never say "information not found" if related concepts exist in the document

    CONTEXT ANALYSIS:
    {context}

    QUESTION: {input}

    ANSWER (be specific and factual):"""

        prompt = ChatPromptTemplate.from_template(prompt_template)
        self.qa_chain = create_stuff_documents_chain(self.llm, prompt)
        print("✅ Context-aware QA ready")

    def _safe_retrieve(self, question: str):
        try:
            if hasattr(self.retriever, 'retrieve'):
                return self.retriever.retrieve(question, k=4)
            else:
                return self.retriever.get_docs(question)
        except Exception as e:
            print(f"Retrieval error: {e}")
            return []

    def answer_question(self, question: str) -> str:
        try:
            docs = self._safe_retrieve(question)
            if not docs:
                return "Unable to find relevant information."
            
            response = self.qa_chain.invoke({
                "input": question,
                "context": docs
            })
            
            return response.strip()
        except Exception as e:
            print(f"Answer error: {e}")
            return f"Error: {str(e)}"

class SimpleRetriever:
    def __init__(self, documents, embedding_model, index):
        self.docs = documents
        self.embedding_model = embedding_model
        self.index = index
        
    def retrieve(self, query: str, k: int = 4):
        try:
            query_embedding = self.embedding_model.encode([query])
            import faiss
            faiss.normalize_L2(query_embedding)
            scores, indices = self.index.search(query_embedding.astype('float32'), k)
            return [self.docs[i] for i in indices[0] if i < len(self.docs)]
        except:
            return self.docs[:k]

class BasicRetriever:
    def __init__(self, documents):
        self.docs = documents
        
    def get_docs(self, query: str):
        return self.docs[:4]
    
class EnhancedHybridRetriever:
    def __init__(self, documents, embedding_model, alpha=0.6):
        self.docs = documents
        self.alpha = alpha
        self.embedding_model = embedding_model
        
        # Medical/Insurance term synonyms for better matching
        self.term_mapping = {
            'cataract': ['lens', 'refractive', 'vision correction', 'intraocular', 'eye surgery'],
            'heart': ['cardiac', 'cardiovascular', 'coronary', 'myocardial'],
            'cancer': ['malignancy', 'tumor', 'oncology', 'neoplasm', 'carcinoma'],
            'mental': ['psychiatric', 'psychological', 'behavioral'],
            'emergency': ['urgent', 'critical', 'ambulatory'],
            'pregnancy': ['maternity', 'prenatal', 'obstetric', 'childbirth'],
            'dental': ['oral', 'tooth', 'periodontal', 'orthodontic'],
            'diabetes': ['diabetic', 'blood sugar', 'insulin', 'glucose']
        }
        
        doc_texts = [doc.page_content for doc in documents]
        tokenized_docs = [self._expand_query(text.lower()).split() for text in doc_texts]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        embeddings = self.embedding_model.encode([d.page_content for d in documents])
        import faiss
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
    
    def _expand_query(self, query):
        """Expand query with medical/insurance synonyms"""
        expanded = query
        for term, synonyms in self.term_mapping.items():
            if term in query.lower():
                expanded += ' ' + ' '.join(synonyms)
        return expanded
        
    def retrieve(self, query: str, k: int = 5):  # Increased k for better context
        expanded_query = self._expand_query(query.lower())
        tokenized_query = expanded_query.split()
        
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Use expanded query for semantic search too
        query_embedding = self.embedding_model.encode([expanded_query])
        import faiss
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding.astype('float32'), k*2)
        
        final_scores = {}
        
        if len(bm25_scores) > 0:
            bm25_max = max(bm25_scores)
            if bm25_max > 0:
                for i, score in enumerate(bm25_scores):
                    final_scores[i] = (1 - self.alpha) * (score / bm25_max)
        
        for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < len(self.docs):
                if idx in final_scores:
                    final_scores[idx] += self.alpha * score
                else:
                    final_scores[idx] = self.alpha * score
        
        top_indices = sorted(final_scores.keys(), key=lambda x: final_scores[x], reverse=True)[:k]
        return [self.docs[i] for i in top_indices]

