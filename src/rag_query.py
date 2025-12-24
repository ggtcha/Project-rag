import os
import re
import ollama
from typing import List, Generator, Dict, Optional
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_community.chat_message_histories import SQLChatMessageHistory

# ============================================================================
# 1. Setup & Configuration 
# ============================================================================
load_dotenv()

DB_CONFIG = {
    "connection": f"postgresql+psycopg2://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DATABASE')}",
    "collection": os.getenv("COLLECTION_NAME")
}

# ============================================================================
# 2. Resources (Embeddings & Vector Store)
# ============================================================================
embeddings = OllamaEmbeddings(
    model="mxbai-embed-large", 
    base_url="http://localhost:11434"
)

vector_store = PGVector(
    collection_name=DB_CONFIG["collection"],
    connection_string=DB_CONFIG["connection"],
    embedding_function=embeddings,
    use_jsonb=True
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5,
        "fetch_k": 15,
        "lambda_mult": 0.7    
    }
)

# ⚡ เพิ่ม num_predict และปรับ temperature
chat_llm = ChatOllama(
    model="llama3.2:1b", 
    temperature=0.5,  # เพิ่มนิดหน่อยเพื่อให้ตอบคำถามทั่วไปได้หลากหลาย
    num_predict=512
)

# ============================================================================
# 3. Utility & Logic
# ============================================================================
def clean_content(text: str) -> str:
    """ลบขยะทางเทคนิคออกจากข้อความ"""
    text = re.sub(r'dtype:\s*\w+|Name:|Unnamed:|\\n|\t', '', text)
    text = re.sub(r'\bNaN\b|\bnan\b|\bNone\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def format_docs(docs: List) -> str:
    if not docs: 
        return "ไม่พบข้อมูลในระบบ"
    formatted = "\n\n".join([clean_content(doc.page_content) for doc in docs])
    return formatted

def analyze_intent(text: str) -> str:
    """ตรวจสอบเจตนา - ปรับให้แม่นยำและรองรับคำถามทั่วไป"""
    text_lower = text.lower().strip()
    
    # ทักทาย
    greetings = ["สวัสดี", "hi", "hello", "ดี", "หวัดดี", "ว่าไง", "ว่ายังไง", "hey"]
    if len(text.split()) <= 5 and any(g in text_lower for g in greetings):
        return "GREETING"
    
    # คำถามเกี่ยวกับตัวบอท
    about_bot = [
        "คุณคือใคร", "คุณคืออะไร", "ทำอะไรได้บ้าง", "ช่วยอะไรได้", 
        "สามารถทำอะไร", "วิธีใช้งาน", "ใช้ยังไง", "คุณชื่ออะไร",
        "who are you", "what can you do", "how to use"
    ]
    if any(q in text_lower for q in about_bot):
        return "ABOUT_BOT"
    
    # เรื่องคลังสินค้า - ต้องมี keyword ที่ชัดเจน
    warehouse_kw = {
        "serial", "sn", "s/n", "model", "รุ่น", "ที่ไหน", "location", 
        "ตำแหน่ง", "asset", "คลัง", "stock", "สถานะ", "status", 
        "ตึก", "อาคาร", "ห้อง", "spare", "อะไหล่", "หา", "ค้นหา",
        "obsolete", "เลิกใช้", "สำรอง", "inventory", "warehouse",
        "part", "รหัส", "code"
    }
    
    # มีรหัสสินค้าในคำถาม (5 ตัวขึ้นไป)
    has_code = bool(re.search(r'[A-Z0-9]{5,}', text, re.IGNORECASE))
    
    # มี keyword คลังสินค้า
    has_warehouse_keyword = any(k in text_lower for k in warehouse_kw)
    
    if has_code or has_warehouse_keyword:
        return "WAREHOUSE_QUERY"
    
    # Default เป็นคำถามทั่วไป (ให้ LLM ตอบเองได้)
    return "GENERAL_KNOWLEDGE"

def expand_query(question: str) -> List[str]:
    """ขยายคำค้นหา"""
    queries = [question]
    codes = re.findall(r'[A-Z0-9]{5,}', question.upper())
    for code in codes[:3]:
        if code not in queries:
            queries.append(code)
    return queries

def analyze_image_with_vision(image_bytes: bytes) -> str:
    """อ่าน Serial/Model จากรูปภาพ"""
    try:
        print("🖼️ กำลังวิเคราะห์รูปภาพ...")
        
        models_to_try = ['llama3.2-vision:latest', 'llava:latest', 'llama3.2:1b']
        
        for model_name in models_to_try:
            try:
                response = ollama.chat(
                    model=model_name,
                    messages=[{
                        'role': 'user',
                        'content': '''Extract ONLY codes/serial numbers from this image.
Rules:
- List codes separated by commas
- No explanations
- If no codes: reply "unknown"

Your answer:''',
                        'images': [image_bytes]
                    }],
                    options={
                        'num_predict': 50,
                        'temperature': 0.1
                    }
                )
                
                result = response['message']['content'].strip()
                result = result.split('\n')[0].strip()
                result = re.sub(r'["\']', '', result)
                
                print(f"✅ Vision Result ({model_name}): {result}")
                
                if result and result.lower() not in ['unknown', 'none', 'n/a', '']:
                    return result
                    
            except Exception as model_error:
                print(f"⚠️ {model_name} failed: {model_error}")
                continue
        
        return "unknown"
        
    except Exception as e:
        print(f"❌ Vision Error: {e}")
        return "unknown"

# ============================================================================
# 4. Context & Chain Setup - ปรับให้รองรับทุกประเภทคำถาม
# ============================================================================
def context_handler(inputs: Dict) -> str:
    """จัดการ Context - รองรับทั้งคำถามคลังและคำถามทั่วไป"""
    question = inputs.get("question", "")
    image_code = inputs.get("image_code", "unknown")
    
    intent = analyze_intent(question)
    
    print(f"🎯 Intent detected: {intent}")
    
    # ถ้าเป็นทักทาย → ไม่ต้องค้น Vector DB
    if intent == "GREETING":
        return "SYSTEM_MODE: GREETING"
    
    # ถ้าถามเกี่ยวกับบอท → ไม่ต้องค้น Vector DB
    if intent == "ABOUT_BOT":
        return "SYSTEM_MODE: ABOUT_BOT"
    
    # ถ้าเป็นคำถามทั่วไป → ไม่ต้องค้น Vector DB
    if intent == "GENERAL_KNOWLEDGE":
        return "SYSTEM_MODE: GENERAL_KNOWLEDGE"
    
    # ถ้าเป็นคำถามคลัง → ค้น Vector DB
    print("🔍 Searching warehouse database...")
    search_query = f"{question} {image_code}" if image_code != "unknown" else question
    
    try:
        queries = expand_query(search_query)
        all_docs = []
        seen_hashes = set()
        
        for query in queries[:3]:
            docs = retriever.invoke(query)
            print(f"   📄 Query '{query}' → {len(docs)} docs")
            
            for doc in docs:
                doc_hash = hash(doc.page_content)
                if doc_hash not in seen_hashes:
                    seen_hashes.add(doc_hash)
                    all_docs.append(doc)
                    
                    if len(all_docs) >= 5:
                        break
            
            if len(all_docs) >= 5:
                break
        
        if not all_docs:
            return f"SYSTEM_MODE: NOT_FOUND | Query: {question}"
        
        print(f"✅ Total docs: {len(all_docs)}")
        formatted_context = format_docs(all_docs[:5])
        
        if len(formatted_context) > 2000:
            formatted_context = formatted_context[:2000] + "\n...(truncated)"
        
        return f"WAREHOUSE_DATA:\n{formatted_context}"
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return f"SYSTEM_MODE: ERROR | {str(e)}"

# ============================================================================
# 5. Prompt & Chain - ปรับให้รองรับทั้งคำถามคลังและคำถามทั่วไป
# ============================================================================
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """คุณคือ **AI Warehouse Assistant** 🤖 ผู้ช่วยคลังสินค้าอัจฉริยะที่เป็นมิตรและช่วยเหลือได้ทุกเรื่อง

🎯 **ความสามารถของคุณ:**
1. 📦 **ค้นหาข้อมูลคลังสินค้า**: Serial Number, Model, ตำแหน่ง, สถานะ
2. 💬 **สนทนาทั่วไป**: ตอบคำถามความรู้ทั่วไป คุยเรื่องอื่นได้
3. 🖼️ **วิเคราะห์รูปภาพ**: อ่านรหัสจากรูปและค้นหาข้อมูล
4. ❓ **ให้คำแนะนำ**: ช่วยเหลือและตอบคำถามทุกประเภท

---

📋 **กฎการตอบคำถาม (สำคัญมาก!):**

**กรณีที่ 1: ถ้า Context = "SYSTEM_MODE: GREETING"**
→ ทักทายกลับอย่างเป็นมิตร แนะนำตัวเองสั้นๆ และบอกว่าพร้อมช่วยเหลือ

**กรณีที่ 2: ถ้า Context = "SYSTEM_MODE: ABOUT_BOT"**
→ แนะนำตัวเองว่าคุณคือ AI Warehouse Assistant และบอกความสามารถทั้ง 4 ข้อข้างบน

**กรณีที่ 3: ถ้า Context = "SYSTEM_MODE: GENERAL_KNOWLEDGE"**
→ ตอบคำถามทั่วไปได้ตามปกติ โดยใช้ความรู้ของคุณเอง (ไม่ต้องมีข้อมูลจาก Context)
→ ตอบแบบกระชับ ชัดเจน เป็นมิตร ใช้ภาษาไทยที่เข้าใจง่าย

**กรณีที่ 4: ถ้า Context ขึ้นต้นด้วย "WAREHOUSE_DATA:"**
→ ตอบโดยใช้ข้อมูลจาก Context เท่านั้น
→ แสดงข้อมูล: Serial Number, Model, Location, Status, ผู้รับผิดชอบ
→ จัดรูปแบบให้อ่านง่าย ใช้อิโมจิประกอบ
→ **ห้าม** แสดง NaN, None, nan, null

**กรณีที่ 5: ถ้า Context = "SYSTEM_MODE: NOT_FOUND"**
→ บอกผู้ใช้ว่า "ไม่พบข้อมูลในระบบคลังสินค้า"
→ แนะนำให้ลองค้นด้วยคำอื่น หรือถามว่าต้องการความช่วยเหลืออย่างอื่นไหม

**กรณีที่ 6: ถ้า Context = "SYSTEM_MODE: ERROR"**
→ บอกว่าเกิดข้อผิดพลาด และแนะนำให้ลองใหม่

---

💬 **สไตล์การตอบ:**
- เป็นมิตร สุภาพ กระชับ ชัดเจน
- ใช้อิโมจิเบาๆ ให้น่าสนใจ (แต่อย่ามากเกินไป)
- ใช้ภาษาไทยที่เป็นกันเอง เข้าใจง่าย
- ถ้าตอบคำถามทั่วไป → ตอบแบบสั้นๆ ไม่เกิน 3-4 ประโยค

---

**Context ปัจจุบัน:**
{context}

---

**หมายเหตุสำคัญ:** 
- ถ้าเป็นคำถามทั่วไปที่ไม่เกี่ยวกับคลังสินค้า คุณสามารถตอบได้เลยโดยไม่ต้องพึ่ง Context
- ถ้าผู้ใช้ถามเรื่องที่คุณไม่แน่ใจ บอกตรงๆ ว่าไม่แน่ใจ แต่พยายามช่วยเหลือให้มากที่สุด
- เป้าหมายคือ: **ช่วยเหลือผู้ใช้ให้ได้มากที่สุด ทั้งเรื่องคลังและเรื่องทั่วไป**"""),
    MessagesPlaceholder(variable_name="history"), 
    ("human", "{question}"),
])

rag_chain = (
    RunnablePassthrough.assign(context=context_handler)
    | rag_prompt
    | chat_llm
    | StrOutputParser()
)

# ============================================================================
# 6. Session History
# ============================================================================
def get_session_history(session_id: str):
    """สร้าง Chat History"""
    if not os.path.exists('data'):
        os.makedirs('data')
    
    history = SQLChatMessageHistory(
        session_id=session_id, 
        connection_string="sqlite:///data/chat_history.db"
    )
    
    # เก็บแค่ 10 ข้อความล่าสุด
    messages = history.messages
    if len(messages) > 10:
        for msg in messages[:-10]:
            history.messages.remove(msg)
    
    return history

chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    get_session_history, 
    input_messages_key="question", 
    history_messages_key="history", 
)

# ============================================================================
# 7. Main Execution
# ============================================================================
def chat_with_warehouse_system(
    user_id: str, 
    prompt: str, 
    image_bytes: Optional[bytes] = None
) -> Generator[str, None, None]:
    """ฟังก์ชันหลัก - รองรับทุกประเภทคำถาม"""
    
    print(f"\n{'='*60}")
    print(f"🔹 User: {user_id} | Prompt: {prompt[:50]}...")
    
    if not prompt or not prompt.strip():
        prompt = "ช่วยหารหัสสินค้าจากรูปภาพ" if image_bytes else "สวัสดีครับ"
    
    image_code = "unknown"
    final_prompt = prompt.strip()

    if image_bytes:
        print("📸 Analyzing image...")
        image_code = analyze_image_with_vision(image_bytes)
        print(f"📋 Code found: {image_code}")
        
        if image_code != "unknown":
            if len(prompt.split()) <= 5:
                final_prompt = f"ตรวจสอบรหัส {image_code}"
            else:
                final_prompt = f"{prompt} (รหัส: {image_code})"
        else:
            final_prompt = f"{prompt} (ไม่พบรหัสชัดเจน)"
    
    print(f"{'='*60}\n")
    
    try:
        has_output = False
        buffer = ""
        chunk_count = 0
        
        for chunk in chain_with_history.stream(
            {"question": final_prompt, "image_code": image_code}, 
            config={"configurable": {"session_id": user_id}}
        ):
            if chunk:
                has_output = True
                buffer += chunk
                chunk_count += 1
                
                if chunk_count % 3 == 0 or len(buffer) > 50:
                    yield buffer
                    buffer = ""
        
        if buffer:
            yield buffer
        
        if not has_output:
            yield "⚠️ ไม่สามารถประมวลผลได้ กรุณาลองใหม่"
            
    except Exception as e:
        error_msg = f"⚠️ เกิดข้อผิดพลาด: {str(e)}"
        print(f"❌ Exception: {error_msg}")
        yield error_msg

def chat_with_lm(user_id: str, prompt: str) -> Generator[str, None, None]:
    """Wrapper สำหรับ backward compatibility"""
    return chat_with_warehouse_system(user_id, prompt, None)