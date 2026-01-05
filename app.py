import streamlit as st
import sys
import os
import time
import json
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage
from src.rag_query import (
    chat_with_warehouse_system,
    get_session_history
)

# =====================================================================
# 📁 1. ส่วนจัดการไฟล์ (ต้องอยู่บนสุดเพื่อป้องกัน Error 'not defined')
# =====================================================================
HISTORY_FILE = "chat_sessions.json"

def load_all_sessions():
    """โหลดรายชื่อ session จากไฟล์ JSON"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

def save_all_sessions(sessions):
    """บันทึกรายชื่อ session ลงไฟล์ JSON"""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(sessions, f, ensure_ascii=False, indent=4)

# =====================================================================
# 🖥️ 2. Page Config & CSS
# =====================================================================
sys.path.append(os.path.abspath(os.getcwd()))

st.set_page_config(
    page_title="AI IT Support Assistant",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def apply_custom_css():
    # โหลดจากไฟล์นอก (ถ้ามี)
    css_file = Path(__file__).parent / "styles.css"
    if css_file.exists():
        with open(css_file, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    # 🎨 แก้ปัญหาตัวหนังสือกลืนพื้นหลัง (Force Contrast)
    st.markdown("""
        <style>
            /* พื้นหลังหน้าหลักต้องขาว/สว่าง และตัวหนังสือต้องดำเข้ม */
            .stApp { background-color: #FFFFFF !important; color: #1A1A1A !important; }
            
            /* ปรับแต่งหัวข้อ */
            h1, h2, h3 { color: #2C3E50 !important; }
            
            /* ส่วนของข้อความแชท */
            .stMarkdown p, .stMarkdown li { 
                color: #1A1A1A !important; 
                font-size: 1.05rem !important;
                line-height: 1.6 !important;
            }
            
            /* Sidebar ให้สีเข้มแต่ตัวหนังสือขาวชัดเจน */
            [data-testid="stSidebar"] { background-color: #1E1E1E !important; }
            [data-testid="stSidebar"] .stMarkdown p { color: #FFFFFF !important; }
            [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #FFFFFF !important; }
            
            /* ปุ่มใน Sidebar */
            [data-testid="stSidebar"] .stButton button {
                background-color: #333333 !important;
                color: #FFFFFF !important;
                border: 1px solid #444 !important;
            }
            [data-testid="stSidebar"] .stButton button:hover {
                border-color: #FF4B4B !important;
                color: #FF4B4B !important;
            }
        </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# =====================================================================
# ⚙️ 3. Session State Initialization
# =====================================================================
if "all_sessions" not in st.session_state:
    st.session_state.all_sessions = load_all_sessions()

if "user_id" not in st.session_state:
    st.session_state.user_id = f"session_{int(time.time())}"

if "chat_history" not in st.session_state:
    db = get_session_history(st.session_state.user_id)
    st.session_state.chat_history = db.messages

if "is_generating" not in st.session_state:
    st.session_state.is_generating = False

if "stop_generating" not in st.session_state:
    st.session_state.stop_generating = False

# =====================================================================
# 🛠️ 4. Sidebar (แสดงประวัติที่โหลดมา)
# =====================================================================
with st.sidebar:
    st.markdown("## 🛠️ เมนู")
    
    if st.button("➕ เริ่มแชทใหม่ (New Chat)", use_container_width=True, type="primary"):
        st.session_state.user_id = f"session_{int(time.time())}"
        st.session_state.chat_history = []
        st.session_state.is_generating = False
        st.rerun()

    st.markdown("---")
    st.markdown("### 🕒 ประวัติการสนทนา")
    
    if not st.session_state.all_sessions:
        st.caption("ยังไม่มีประวัติการสนทนา")
    else:
        for chat in st.session_state.all_sessions:
            # ตรวจสอบว่าเป็นแชทที่เลือกอยู่หรือไม่
            is_active = chat['id'] == st.session_state.user_id
            label = f"💬 {chat['title']}"
            if st.button(label, key=f"hist_{chat['id']}", use_container_width=True):
                st.session_state.user_id = chat['id']
                db = get_session_history(chat['id'])
                st.session_state.chat_history = db.messages
                st.rerun()

    st.markdown("---")
    st.markdown("### 💡 คำถามตัวอย่าง")
    example_questions = ["Serial CN43KR3017 คืออะไร?", "มี ThinkPad กี่เครื่อง?", "อุปกรณ์ Spare มีอะไรบ้าง?"]
    for q in example_questions:
        if st.button(q, use_container_width=True, key=f"ex_{q}", disabled=st.session_state.is_generating):
            st.session_state.selected_question = q

# =====================================================================
# 💬 5. Chat Area & Input
# =====================================================================
st.markdown("<h1 style='text-align: center;'>🖥️ AI IT Support Assistant</h1>", unsafe_allow_html=True)
st.markdown("---")

chat_container = st.container()

with chat_container:
    if not st.session_state.chat_history:
        st.markdown("<div style='text-align: center; padding: 40px;'><h3>👋 สวัสดีครับ! มีอะไรให้ช่วยไหมจ๊ะ?</h3></div>", unsafe_allow_html=True)
    
    for msg in st.session_state.chat_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role, avatar="👤" if role == "user" else "🤖"):
            st.markdown(msg.content)

# ปุ่ม Stop
if st.session_state.is_generating:
    if st.button("⏹️ หยุดการตอบ (Stop)", use_container_width=True):
        st.session_state.stop_generating = True
        st.rerun()

# จัดการ Input
if "selected_question" in st.session_state and not st.session_state.is_generating:
    prompt = st.session_state.selected_question
    del st.session_state.selected_question
else:
    prompt = st.chat_input("💬 พิมพ์คำถามของคุณที่นี่...", disabled=st.session_state.is_generating)

# =====================================================================
# 🧠 6. Logic การประมวลผล
# =====================================================================
if prompt:
    st.session_state.is_generating = True
    st.session_state.stop_generating = False
    
    # --- บันทึกประวัติใหม่ลง JSON ทันที ---
    session_exists = any(s['id'] == st.session_state.user_id for s in st.session_state.all_sessions)
    if not session_exists:
        title = prompt[:30] + "..." if len(prompt) > 30 else prompt
        st.session_state.all_sessions.insert(0, {"id": st.session_state.user_id, "title": title})
        save_all_sessions(st.session_state.all_sessions)

    with chat_container:
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar="🤖"):
            response_placeholder = st.empty()
    
    full_text = ""
    try:
        for chunk in chat_with_warehouse_system(st.session_state.user_id, prompt):
            if st.session_state.stop_generating:
                full_text += "\n\n⚠️ _[หยุดการตอบโดยผู้ใช้]_"
                break
            full_text += chunk
            response_placeholder.markdown(full_text) 
        
        st.session_state.chat_history.append(HumanMessage(content=prompt))
        st.session_state.chat_history.append(AIMessage(content=full_text))

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาด: {str(e)}")
    finally:
        st.session_state.is_generating = False
        st.rerun()

st.markdown("---")
st.markdown("<div style='text-align: center; color: #95a5a6;'>🔒 ข้อมูลปลอดภัย | Powered by AI & RAG</div>", unsafe_allow_html=True)