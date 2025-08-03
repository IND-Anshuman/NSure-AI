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
    section_patterns = [
        r'(?i)(coverage|benefit|exclusion|limitation|condition|procedure|treatment)',
        r'(?i)(section|clause|article|paragraph)\s+\d+',
        r'(?i)(what.*covered|what.*not.*covered|how.*works)',
        r'\d+\.\s+[A-Z]',
        r'[A-Z][a-z]+:\s*',
    ]
    
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
            bm25_max = max(bm25_scores) if max(bm25_scores) > 0 else 1
            for i, score in enumerate(bm25_scores):
                final_scores[i] = self.alpha * (score / bm25_max)
        
        for i, (semantic_score, doc_idx) in enumerate(zip(scores[0], indices[0])):
            if doc_idx < len(self.docs):
                if doc_idx in final_scores:
                    final_scores[doc_idx] += (1 - self.alpha) * semantic_score
                else:
                    final_scores[doc_idx] = (1 - self.alpha) * semantic_score
        
        top_docs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [self.docs[idx] for idx, _ in top_docs]

class OptimizedRAGCore:
    def __init__(self, embedding_model, llm):
        self.embedding_model = embedding_model
        self.llm = llm
        
        # Optimized prompt for exact format
        self.prompt_template = ChatPromptTemplate.from_template("""
You are an insurance policy expert. Answer questions using ONLY the provided context.

CRITICAL INSTRUCTIONS:
- Provide direct, complete answers exactly as written in the policy
- Start answers with specific details (grace period of X days, waiting period of X months, etc.)
- Use the EXACT wording from the policy document
- Include all relevant conditions and limits mentioned
- Do NOT summarize or paraphrase - copy the exact language
- Do NOT add explanations or context beyond what's asked

Context: {context}

Question: {question}

Answer (use exact policy language):""")

    async def process_queries(self, pdf_url: str, questions: List[str]) -> List[str]:
        try:
            text = get_pdf_text_from_url(pdf_url)
            if not text:
                return ["Error: Could not extract text from PDF"] * len(questions)
            
            chunks = context_aware_chunk_text(text)
            docs = [Document(page_content=chunk) for chunk in chunks]
            
            retriever = HybridRetriever(docs, self.embedding_model)
            
            answers = []
            for question in questions:
                relevant_docs = retriever.retrieve(question, k=3)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                
                response = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    self._get_answer, 
                    context, 
                    question
                )
                
                # Clean and format response
                clean_answer = self._format_answer(response)
                answers.append(clean_answer)
            
            return answers
            
        except Exception as e:
            return [f"Error processing query: {str(e)}"] * len(questions)
    
    def _get_answer(self, context: str, question: str) -> str:
        messages = self.prompt_template.format_messages(
            context=context,
            question=question
        )
        
        response = self.llm.invoke(messages)
        return response.content.strip()
    
    def _format_answer(self, raw_answer: str) -> str:
        # Remove common LLM prefixes and suffixes
        prefixes_to_remove = [
            "According to the policy,",
            "The policy states that",
            "Based on the document,",
            "The document mentions that",
            "Answer:",
            "Response:",
        ]
        
        answer = raw_answer.strip()
        
        for prefix in prefixes_to_remove:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()
        
        # Ensure proper capitalization and punctuation
        if answer and not answer[0].isupper():
            answer = answer[0].upper() + answer[1:]
        
        if answer and not answer.endswith('.'):
            answer += '.'
        
        return answer