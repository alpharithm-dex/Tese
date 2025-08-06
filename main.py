# main.py - Enhanced Competitive Version
import logging
import os
import time
import sqlite3
from typing import List, Dict
from functools import lru_cache

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import ollama
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- Enhanced Configuration ---
# Setup advanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("edusense_hub.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- MODEL CONFIGURATION ---
# Multi-model strategy for different scenarios
EMBEDDING_MODEL = "nomic-embed-text"  # Lightweight but powerful embeddings
PRIMARY_CHAT_MODEL = "gemma3:7b-it-q4_K_M"  # Balanced power/performance
FALLBACK_CHAT_MODEL = "gemma3:1b-it-q4_K_M"  # For low-power situations
VISION_MODEL = "llava:latest"  # For image-based questions

# --- FILE/PATH CONFIGURATION ---
SYLLABUS_FILE = "syllabus.txt"
VECTOR_STORE_PATH = "faiss_index"
CACHE_DB = "query_cache.db"
POWER_LOG = "power_usage.log"

# --- Performance Constants ---
MAX_CONTEXT_TOKENS = 2048  # Based on model capacity
MAX_RESPONSE_TOKENS = 256
TEMPERATURE = 0.3  # Balance creativity vs accuracy
TIMEOUT_SECONDS = 15  # Response timeout

# --- Advanced Prompt Engineering ---
PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """
    You are Tese, a friendly AI Tutor for students.
    Strictly use ONLY the provided syllabus context to answer.
    If the answer isn't in the context, politely decline to answer.
    Respond in simple English suitable for O-Level students.
    Format responses with bullet points when explaining concepts.
    Always cite the source chapter and page if available.
    """),
    ("human", """
    Syllabus Context:
    {context}

    Student Question:
    {question}
    """)
])

# --- FastAPI Application Setup ---
app = FastAPI(
    title="Tese Hub API",
    description="Competitive AI Tutor backend for offline learning",
    version="2.0.0",
    docs_url="/docs",
    redoc_url=None
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Models ---
class AskRequest(BaseModel):
    question: str
    image_data: str = None  # Base64 encoded image for multimodal queries

class AskResponse(BaseModel):
    answer: str
    sources: List[Dict[str, str]]  # Source documents with metadata
    model_used: str
    response_time: float
    power_used: float = None  # In watt-seconds

# --- Global State ---
retriever = None
power_monitor_active = False
current_power_mode = "normal"  # normal | low_power

# --- Power Management System ---
def log_power_usage(watt_seconds: float):
    """Log power consumption for efficiency tracking"""
    with open(POWER_LOG, "a") as f:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp},{watt_seconds}\n")

def calculate_power_usage(start_time: float, end_time: float) -> float:
    """Calculate power consumption based on processing time"""
    duration = end_time - start_time
    # Power profiles based on mode (normal: 5W, low_power: 2W)
    power_watts = 2 if current_power_mode == "low_power" else 5
    return power_watts * duration

def switch_power_mode(battery_level: int):
    """Switch power modes based on available battery"""
    global current_power_mode
    if battery_level < 20:
        current_power_mode = "low_power"
        logger.warning(f"Switched to LOW POWER mode. Battery: {battery_level}%")
    else:
        current_power_mode = "normal"

# --- Cache System ---
def init_cache_db():
    """Initialize query cache database"""
    with sqlite3.connect(CACHE_DB) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS query_cache (
            question TEXT PRIMARY KEY,
            answer TEXT,
            sources TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)

@lru_cache(maxsize=500)
def get_cached_response(question: str):
    """Check for cached response"""
    try:
        with sqlite3.connect(CACHE_DB) as conn:
            cursor = conn.execute(
                "SELECT answer, sources FROM query_cache WHERE question = ?",
                (question,)
            )
            result = cursor.fetchone()
            return result if result else None
    except sqlite3.Error as e:
        logger.error(f"Cache error: {str(e)}")
        return None

def cache_response(question: str, answer: str, sources: str):
    """Cache new response"""
    try:
        with sqlite3.connect(CACHE_DB) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO query_cache (question, answer, sources) VALUES (?, ?, ?)",
                (question, answer, sources)
            )
    except sqlite3.Error as e:
        logger.error(f"Cache update error: {str(e)}")

# --- Enhanced RAG Pipeline ---
def initialize_retriever():
    """Advanced syllabus processing with metadata extraction"""
    global retriever
    
    logger.info("Initializing enhanced RAG pipeline...")
    start_time = time.time()
    
    try:
        # Load syllabus with metadata extraction
        loader = TextLoader(SYLLABUS_FILE)
        docs = loader.load()
        
        # Advanced text splitting with chapter awareness
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", "Chapter", "Section", "•"],
            length_function=len
        )
        documents = text_splitter.split_documents(docs)
        
        # Inject metadata based on content
        for i, doc in enumerate(documents):
            doc.metadata = {
                "doc_id": f"doc_{i}",
                "source": SYLLABUS_FILE,
                "chapter": extract_chapter(doc.page_content),
                "importance": calculate_importance(doc.page_content)
            }
        
        # Create optimized embeddings
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        
        # Create FAISS index with metadata storage
        vector_store = FAISS.from_documents(
            documents, 
            embeddings,
            normalize_L2=True  # For better cosine similarity
        )
        
        # Configure advanced retriever
        retriever = vector_store.as_retriever(
            search_type="mmr",  # Maximal Marginal Relevance
            search_kwargs={
                'k': 5,  # Number of docs to retrieve
                'lambda_mult': 0.5,  # Diversity vs relevance balance
                'score_threshold': 0.7  # Minimum similarity score
            }
        )
        
        duration = time.time() - start_time
        logger.info(f"RAG pipeline initialized in {duration:.2f}s")
        
    except Exception as e:
        logger.critical(f"RAG initialization failed: {str(e)}")
        raise

def extract_chapter(text: str) -> str:
    """Extract chapter information from text snippet"""
    # Simplified chapter detection - implement ML-based in production
    if "chapter" in text.lower():
        return text.split()[0]
    return "General"

def calculate_importance(text: str) -> float:
    """Calculate importance score for syllabus content"""
    # Placeholder - implement TF-IDF or ML-based scoring
    key_terms = ["key concept", "important", "exam focus", "essential"]
    return 1.0 if any(term in text.lower() for term in key_terms) else 0.5

def rerank_documents(query: str, documents: List) -> List:
    """Re-rank documents based on advanced similarity"""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    query_embedding = embeddings.embed_query(query)
    doc_embeddings = [embeddings.embed_query(doc.page_content) for doc in documents]
    
    # Calculate cosine similarities
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    
    # Combine with original scores and importance
    for i, doc in enumerate(documents):
        combined_score = (
            0.6 * similarities[i] + 
            0.3 * doc.metadata.get('score', 0) +
            0.1 * doc.metadata.get('importance', 0.5)
        )
        doc.metadata['combined_score'] = combined_score
    
    # Sort by combined score
    return sorted(documents, key=lambda x: x.metadata['combined_score'], reverse=True)

# --- API Endpoints ---
@app.on_event("startup")
def startup_event():
    """Initialize systems on startup"""
    logger.info("EduSense Hub starting...")
    init_cache_db()
    initialize_retriever()
    logger.info("Startup complete. Ready for requests.")

@app.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest, 
    background_tasks: BackgroundTasks
):
    """Enhanced question answering endpoint"""
    start_time = time.time()
    question = request.question.strip()
    
    # Check for cached response
    if cached := get_cached_response(question):
        answer, sources = cached
        return AskResponse(
            answer=answer,
            sources=eval(sources),
            model_used="cache",
            response_time=time.time() - start_time
        )
    
    # Validate retriever state
    if not retriever:
        raise HTTPException(503, "System initializing. Try again shortly.")
    
    try:
        # Retrieve context
        relevant_docs = retriever.invoke(question)
        reranked_docs = rerank_documents(question, relevant_docs)
        
        # Select optimal context length
        context_str = build_context(reranked_docs, MAX_CONTEXT_TOKENS)
        
        # Handle multimodal queries
        if request.image_data:
            return handle_multimodal_query(question, request.image_data, start_time)
        
        # Generate response with fallback mechanism
        response, model_used = generate_response(question, context_str)
        
        # Extract and format sources
        sources = format_sources(reranked_docs)
        
        # Cache response
        cache_response(question, response, str(sources))
        
        # Calculate power usage
        end_time = time.time()
        power_used = calculate_power_usage(start_time, end_time)
        background_tasks.add_task(log_power_usage, power_used)
        
        return AskResponse(
            answer=response,
            sources=sources,
            model_used=model_used,
            response_time=end_time - start_time,
            power_used=power_used
        )
        
    except TimeoutError:
        logger.warning(f"Timeout processing: {question}")
        return JSONResponse(
            status_code=504,
            content={"error": "Processing timeout. Simplify your question."}
        )
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal processing error. Try again later."}
        )

# --- Enhanced Processing Functions ---
def build_context(documents: List, max_tokens: int) -> str:
    """Build context within token limits"""
    context = []
    token_count = 0
    
    for doc in documents:
        doc_content = f"SOURCE: {doc.metadata.get('chapter', '')}\n{doc.page_content}"
        doc_tokens = len(doc_content.split())
        
        if token_count + doc_tokens <= max_tokens:
            context.append(doc_content)
            token_count += doc_tokens
        else:
            break
    
    return "\n\n".join(context)

def generate_response(question: str, context: str) -> (str, str):
    """Generate response with model fallback"""
    model_used = PRIMARY_CHAT_MODEL
    
    try:
        # Try primary model first
        chain = PROMPT_TEMPLATE | ollama.Ollama(
            model=PRIMARY_CHAT_MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_RESPONSE_TOKENS,
            timeout=TIMEOUT_SECONDS
        ) | StrOutputParser()
        
        response = chain.invoke({
            "question": question,
            "context": context
        })
        
        # Validate response quality
        if is_low_quality(response):
            raise ValueError("Low quality response detected")
            
        return response, PRIMARY_CHAT_MODEL
        
    except Exception as e:
        logger.warning(f"Primary model failed ({str(e)}), using fallback")
        try:
            # Fallback to lightweight model
            chain = PROMPT_TEMPLATE | ollama.Ollama(
                model=FALLBACK_CHAT_MODEL,
                temperature=TEMPERATURE,
                max_tokens=MAX_RESPONSE_TOKENS,
                timeout=TIMEOUT_SECONDS
            ) | StrOutputParser()
            
            return chain.invoke({
                "question": question,
                "context": context
            }), FALLBACK_CHAT_MODEL
        except Exception as fallback_e:
            logger.critical(f"Fallback model failed: {str(fallback_e)}")
            return "I'm having trouble answering that. Please try again later.", "error"

def handle_multimodal_query(question: str, image_data: str, start_time: float) -> AskResponse:
    """Process image-based queries using LLaVA"""
    try:
        # Extract text from image
        response = ollama.generate(
            model=VISION_MODEL,
            prompt=f"Describe this image in detail: {image_data}",
            timeout=TIMEOUT_SECONDS
        )
        
        # Combine visual description with question
        enhanced_question = f"{question} about this: {response['response']}"
        
        # Retrieve relevant context
        relevant_docs = retriever.invoke(enhanced_question)
        reranked_docs = rerank_documents(enhanced_question, relevant_docs)
        context_str = build_context(reranked_docs, MAX_CONTEXT_TOKENS)
        
        # Generate final response
        chain = PROMPT_TEMPLATE | ollama.Ollama(
            model=PRIMARY_CHAT_MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_RESPONSE_TOKENS,
            timeout=TIMEOUT_SECONDS
        ) | StrOutputParser()
        
        response = chain.invoke({
            "question": question,
            "context": context_str
        })
        
        end_time = time.time()
        power_used = calculate_power_usage(start_time, end_time)
        
        return AskResponse(
            answer=response,
            sources=format_sources(reranked_docs),
            model_used=f"{VISION_MODEL}+{PRIMARY_CHAT_MODEL}",
            response_time=end_time - start_time,
            power_used=power_used
        )
        
    except Exception as e:
        logger.error(f"Multimodal processing failed: {str(e)}")
        return AskResponse(
            answer="I couldn't process your image. Try a text-only question.",
            sources=[],
            model_used="error",
            response_time=time.time() - start_time
        )

def format_sources(documents: List) -> List[Dict[str, str]]:
    """Format source documents for response"""
    return [{
        "content": doc.page_content,
        "chapter": doc.metadata.get("chapter", "Unknown"),
        "score": f"{doc.metadata.get('combined_score', 0):.2f}",
        "importance": doc.metadata.get("importance", 0.5)
    } for doc in documents]

def is_low_quality(response: str) -> bool:
    """Detect low-quality responses"""
    low_quality_phrases = [
        "I don't know",
        "not in the context",
        "sorry",
        "unable to answer"
    ]
    
    return any(phrase in response.lower() for phrase in low_quality_phrases)

# --- System Monitoring Endpoint ---
@app.get("/system-status")
def system_status():
    """Endpoint for monitoring system health"""
    return {
        "status": "operational" if retriever else "initializing",
        "model": PRIMARY_CHAT_MODEL,
        "power_mode": current_power_mode,
        "cache_size": len(get_cached_response.cache_info()),
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
    }

# --- Syllabus Update Endpoint ---
@app.post("/update-syllabus")
def update_syllabus(file_path: str):
    """Hot-reload syllabus content"""
    global SYLLABUS_FILE, retriever
    try:
        SYLLABUS_FILE = file_path
        initialize_retriever()
        return {"status": "success", "message": "Syllabus updated"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )
