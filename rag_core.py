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

def context_aware_chunk_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 200) -> List[str]:
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
        
    def retrieve(self, query: str, k: int = 5):
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
        
        self.prompt_template = ChatPromptTemplate.from_template("""
Answer the question with complete information from the policy document. Provide comprehensive details including all conditions, timeframes, amounts, and requirements mentioned.

Examples of good answers:
- "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits."
- "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered."

Write complete sentences with full details. Include specific numbers, timeframes, conditions, and amounts.

Context: {context}

Question: {question}

Complete Answer:""")

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
                relevant_docs = retriever.retrieve(question, k=5)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                
                response = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    self._get_answer, 
                    context, 
                    question
                )
                
                clean_answer = self._format_answer(response, question)
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
    
    def _format_answer(self, raw_answer: str, question: str) -> str:
        answer = raw_answer.strip()
        
        prefixes_to_remove = [
            "According to the policy,",
            "The policy states that",
            "Based on the document,",
            "The document mentions that",
            "Answer:",
            "Response:",
            "Complete Answer:",
        ]
        
        for prefix in prefixes_to_remove:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()
        
        if not answer:
            return f"Information not found in the policy document for: {question}"
        
        if answer.startswith('[') and 'cut off' in answer.lower():
            return f"Information not available in the provided policy text."
        
        if answer.lower().startswith('this question cannot be answered'):
            return f"Information not available in the provided policy text."
        
        if 'provided text does not' in answer.lower():
            return f"Information not available in the provided policy text."
        
        if answer and not answer[0].isupper():
            answer = answer[0].upper() + answer[1:]
        
        if answer and not answer.endswith('.'):
            answer += '.'
        
        answer = re.sub(r'\s+', ' ', answer)
        answer = answer.replace('â', '-')
        answer = answer.replace('*', '')
        
        return answer