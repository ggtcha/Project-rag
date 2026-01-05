from typing import Generator
import os
import re
from dotenv import load_dotenv
from functools import lru_cache

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_community.chat_message_histories import PostgresChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# ============================================================================
# 1. Setup & Configuration
# ============================================================================
load_dotenv()

PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_HOST = os.getenv("PG_HOST")
PG_PORT = os.getenv("PG_PORT")
PG_DATABASE = os.getenv("PG_DATABASE")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")

# 🔽 ลด RAM (เปลี่ยนจาก large)
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

OLLAMA_BASE_URL = "http://localhost:11434"

# ============================================================================
# 2. Database Connection Strings
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
# 3. Lazy Initialization
# ============================================================================
_embeddings = None
_vectorstore = None
_retriever = None
_llm = None

# ============================================================================
# 4. Embeddings / VectorStore / Retriever
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
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.4
            }
        )
    return _retriever

# ============================================================================
# 5. LLM
# ============================================================================

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOllama(
            model=LLM_MODEL,
            temperature=0.0,
            stream=True,
            base_url=OLLAMA_BASE_URL,
            num_ctx=1024,
            num_predict=256
        )
    return _llm

# ============================================================================
# 6. Prompt Templates
# ============================================================================

WAREHOUSE_PROMPT = ChatPromptTemplate.from_template("""
คุณคือ AI ผู้ช่วยคลังสินค้า

กฎสำคัญ:
- ตอบโดยอ้างอิงจากข้อมูลใน CONTEXT เท่านั้น
- ห้ามเดา ห้ามใช้ความรู้ภายนอก
- ถ้าไม่มีข้อมูล ให้ตอบว่า "ไม่พบข้อมูลในระบบ"

CONTEXT:
{context}

คำถาม:
{question}

คำตอบ (ภาษาไทย ชัดเจน กระชับ):
""")

GENERAL_PROMPT = ChatPromptTemplate.from_template("""
คุณคือ AI ผู้ช่วยอเนกประสงค์

สามารถตอบ:
- ความรู้ทั่วไป
- ประวัติศาสตร์
- ตรรกะ
- การอธิบายแนวคิด
- How-to

คำถาม:
{question}

คำตอบ (ภาษาไทย เข้าใจง่าย):
""")

# ============================================================================
# 7. Chat History
# ============================================================================

@lru_cache(maxsize=10)
def get_session_history(session_id: str):
    return PostgresChatMessageHistory(
        connection_string=PSYCOPG_CONN_INFO,
        session_id=session_id
    )

# ============================================================================
# 8. Intent Classification
# ============================================================================

WAREHOUSE_KEYWORDS = [
    "serial", "asset", "รุ่น", "สินค้า",
    "คลัง", "ตำแหน่ง", "สถานะ", "หมายเลข"
]

SERIAL_PATTERN = r"[A-Z0-9]{8,}"

def classify_intent(question: str) -> str:
    q = question.lower()

    if any(k in q for k in WAREHOUSE_KEYWORDS):
        return "warehouse"

    if re.search(SERIAL_PATTERN, question):
        return "warehouse"

    return "general"

# ============================================================================
# 9. Context Compression
# ============================================================================

def compress_context(docs, max_chars: int = 2000) -> str:
    text = ""
    for doc in docs:
        if len(text) + len(doc.page_content) > max_chars:
            break
        text += doc.page_content.strip() + "\n---\n"
    return text.strip()

# ============================================================================
# 10. Main Chat Function
# ============================================================================
def chat_with_warehouse_system(
    session_id: str,
    question: str,
    image: bytes | None = None
) -> Generator[str, None, None]:

    llm = get_llm()
    history = get_session_history(session_id)

    intent = classify_intent(question)

    # =========================
    # STEP 1: Try Retrieval (only if intent suggests warehouse)
    # =========================
    docs = []
    context = ""

    if intent == "warehouse":
        retriever = get_retriever()
        try:
            docs = retriever.invoke(question)
            context = compress_context(docs) if docs else ""
        except Exception as e:
            print(f"[DEBUG] Retrieval error: {e}")
            docs = []
            context = ""

    # =========================
    # STEP 2: Decide Mode by CONTEXT (not intent)
    # =========================
    use_rag = bool(context and len(context) > 50)

    # =========================
    # GENERAL MODE (Fallback / No RAG)
    # =========================
    if not use_rag:
        chain = (
            {"question": RunnablePassthrough()}
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
    # WAREHOUSE MODE (RAG)
    # =========================
    chain = (
        {
            "context": lambda _: context,
            "question": RunnablePassthrough()
        }
        | WAREHOUSE_PROMPT
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
# 11. Utilities
# ============================================================================

def clear_session_history(session_id: str):
    history = get_session_history(session_id)
    history.clear()
    get_session_history.cache_clear()

def cleanup_resources():
    global _vectorstore, _embeddings, _llm
    _vectorstore = None
    _embeddings = None
    _llm = None
    get_session_history.cache_clear()
