from typing import Generator
import os
import re
from dotenv import load_dotenv
from functools import lru_cache
from datetime import datetime

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_community.chat_message_histories import PostgresChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# ============================================================================
# Configuration
# ============================================================================
PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_HOST = os.getenv("PG_HOST")
PG_PORT = os.getenv("PG_PORT")
PG_DATABASE = os.getenv("PG_DATABASE")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = "http://localhost:11434"

# ============================================================================
# Database Connection
# ============================================================================
SQLALCHEMY_DB_URL = (
    f"postgresql+psycopg2://"
    f"{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"
)

PSYCOPG_CONN_INFO = (
    f"dbname={PG_DATABASE} "
    f"user={PG_USER} "
    f"password={PG_PASSWORD} "
    f"host={PG_HOST} "
    f"port={PG_PORT}"
)

# ============================================================================
# Lazy Initialization
# ============================================================================
_embeddings = None
_vectorstore = None
_retriever = None
_llm = None

# ============================================================================
# Core Functions
# ============================================================================

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = OllamaEmbeddings(
            model=EMBED_MODEL,
            base_url=OLLAMA_BASE_URL
        )
    return _embeddings

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = PGVector(
            connection_string=SQLALCHEMY_DB_URL,
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
        )
    return _vectorstore

def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = get_vectorstore().as_retriever(
            search_type="similarity",
            search_kwargs={
                "k":50,  # เพิ่มเป็น 7 เพื่อหาข้อมูลได้หลากหลาย
            }
        )
    return _retriever

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOllama(
            model=LLM_MODEL,
            temperature=0.0,
            stream=True,
            base_url=OLLAMA_BASE_URL,
            num_ctx=4096,
            num_predict=1024
        )
    return _llm

# ============================================================================
# Prompts
# ============================================================================

IT_ASSET_PROMPT = ChatPromptTemplate.from_template("""
คุณคือ AI IT Support Assistant ที่เชี่ยวชาญในการจัดการ IT 

ความสามารถของคุณ:
✅ ค้นหาข้อมูล Asset จาก Serial Number, Model, Asset Number
✅ ตรวจสอบสถานะและตำแหน่งของอุปกรณ์
✅ บอกรายละเอียดสเปค อายุการใช้งาน วันที่จัดซื้อ
✅ แยกประเภทอุปกรณ์ (Laptop, Desktop, Router, Switch, etc.)
✅ นับจำนวนอุปกรณ์ตามเงื่อนไขต่างๆ

กฎการตอบคำถาม:
1. ตอบจากข้อมูลใน CONTEXT เท่านั้น - ห้ามเดา ห้ามสมมติ
2. ถ้าคำถามถามหา Serial/Asset ที่ระบุชัดเจน ให้ตอบรายละเอียดครบถ้วน
3. ถ้าถามนับจำนวน ให้ตอบตัวเลขชัดเจน และระบุรายการถ้าไม่เยอะ
4. ถ้าถามตำแหน่ง ให้ตอบ Location ชัดเจน
5. ถ้าไม่มีข้อมูล ให้ตอบว่า "ไม่พบข้อมูลในระบบ"
6. จัดรูปแบบคำตอบให้อ่านง่าย ใช้ bullet points หรือตารางถ้าเหมาะสม

วันที่ปัจจุบัน: {current_date}

CONTEXT จากฐานข้อมูล:
{context}

คำถามจากผู้ใช้:
{question}

คำตอบ (ภาษาไทย ชัดเจน เป็นระเบียบ):
""")

GENERAL_PROMPT = ChatPromptTemplate.from_template("""
คุณคือ AI IT Support Assistant ที่เป็นมิตรและช่วยเหลือ

คุณสามารถ:
- ตอบคำถามทั่วไปเกี่ยวกับไอที
- ให้คำแนะนำการแก้ปัญหาเบื้องต้น
- อธิบายเทคโนโลยีและแนวคิดต่างๆ
- สนทนาเป็นกันเอง

วันที่ปัจจุบัน: {current_date}

คำถาม:
{question}

คำตอบ (ภาษาไทย เป็นกันเอง):
""")

# ============================================================================
# Chat History
# ============================================================================

@lru_cache(maxsize=10)
def get_session_history(session_id: str):
    return PostgresChatMessageHistory(
        connection_string=PSYCOPG_CONN_INFO,
        session_id=session_id
    )

# ============================================================================
# Intent Classification
# ============================================================================

IT_ASSET_KEYWORDS = [
    # Asset & Serial
    "serial", "s/n", "asset", "รหัสทรัพย์สิน", "หมายเลขทรัพย์สิน",
    "ซีเรียล", "เลขทรัพย์สิน","serial number","Serialnumber","serialnumber",
    "Serial"
    
    # Device Types
    "thinkpad", "laptop", "notebook", "computer", "คอม", "เครื่อง",
    "desktop", "workstation", "mac mini",
    "switch", "router", "access point", "wifi",
    "printer", "เครื่องพิมพ์",
    
    # Locations
    "ตำแหน่ง", "location", "อยู่ที่", "สถานที่", "ที่ไหน", "where",
    "sriracha", "ศรีราชา", "chonburi", "ชลบุรี",
    "custom server room", "customs building", "kp 4.0",
    
    # Status & Condition
    "สถานะ", "status", "spare", "obsolete", 
    "พร้อมใช้", "available", "เลิกใช้", "เสื่อม",
    "deployable", "deployed",
    
    # Queries
    "มี", "เหลือ", "กี่", "จำนวน", "how many", "count",
    "ค้นหา", "หา", "search", "find", "ตรวจสอบ", "check",
    "รุ่น", "model", "รายการ", "list",
    
    # Purchase & Order
    "จัดซื้อ", "purchased", "order", "po", "ใบสั่งซื้อ",
]

# Pattern สำหรับ Serial, Asset, Order Number
SERIAL_PATTERN = r"[A-Z0-9]{8,}"
ASSET_PATTERN = r"\d{7,8}"
ORDER_PATTERN = r"\d{9,}"

def classify_intent(question: str) -> str:
    """จำแนกประเภทคำถาม"""
    
    q = question.lower()
    
    # เช็ค keywords
    if any(k in q for k in IT_ASSET_KEYWORDS):
        return "it_asset"
    
    # เช็ค patterns
    if re.search(SERIAL_PATTERN, question):
        return "it_asset"
    
    if re.search(ASSET_PATTERN, question):
        return "it_asset"
    
    if re.search(ORDER_PATTERN, question):
        return "it_asset"
    
    return "general"

# ============================================================================
# Context Processing
# ============================================================================
def compress_context(docs, max_chars: int = 15000) -> str:
    if not docs:
        return ""
    
    text = ""
    for i, doc in enumerate(docs, 1):
        doc_text = f"[รายการที่ {i}]\n{doc.page_content.strip()}\n\n"
        
        if len(text) + len(doc_text) > max_chars:
            break
        
        text += doc_text
    
    return text.strip()

# ============================================================================
# Main Chat Function
# ============================================================================

def chat_with_warehouse_system(
    session_id: str,
    question: str,
    image: bytes | None = None
) -> Generator[str, None, None]:
    """ฟังก์ชันหลักสำหรับตอบคำถาม"""
    
    llm = get_llm()
    history = get_session_history(session_id)
    
    # จำแนกความตั้งใจ
    intent = classify_intent(question)
    
    # วันที่ปัจจุบัน
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # =========================
    # STEP 1: Retrieval
    # =========================
    docs = []
    context = ""
    
    if intent == "it_asset":
        retriever = get_retriever()
        try:
            docs = retriever.invoke(question)
            context = compress_context(docs) if docs else ""
            
            # Debug
            print(f"[DEBUG] Found {len(docs)} documents")
            if docs:
                print(f"[DEBUG] Top doc score: {docs[0].metadata.get('score', 'N/A')}")
                
        except Exception as e:
            print(f"[DEBUG] Retrieval error: {e}")
            docs = []
            context = ""
    
    # =========================
    # STEP 2: Choose Mode
    # =========================
    use_rag = bool(context and len(context) > 50)
    
    # =========================
    # GENERAL MODE
    # =========================
    if not use_rag:
        chain = (
            {
                "question": RunnablePassthrough(),
                "current_date": lambda _: current_date
            }
            | GENERAL_PROMPT
            | llm
        )
        
        full_response = ""
        for chunk in chain.stream(question):
            content = getattr(chunk, "content", str(chunk))
            full_response += content
            yield content
        
        history.add_user_message(question)
        history.add_ai_message(full_response)
        return
    
    # =========================
    # IT ASSET MODE (RAG)
    # =========================
    chain = (
        {
            "context": lambda _: context,
            "question": RunnablePassthrough(),
            "current_date": lambda _: current_date
        }
        | IT_ASSET_PROMPT
        | llm
    )
    
    full_response = ""
    for chunk in chain.stream(question):
        content = getattr(chunk, "content", str(chunk))
        full_response += content
        yield content
    
    history.add_user_message(question)
    history.add_ai_message(full_response)

# ============================================================================
# Utilities
# ============================================================================

def clear_session_history(session_id: str):
    """ล้างประวัติการสนทนา"""
    history = get_session_history(session_id)
    history.clear()
    get_session_history.cache_clear()

def cleanup_resources():
    """ปิดและล้าง resources"""
    global _vectorstore, _embeddings, _llm, _retriever
    _vectorstore = None
    _embeddings = None
    _llm = None
    _retriever = None
    get_session_history.cache_clear()