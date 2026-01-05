import streamlit as st
import sys
import os
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage
from src.rag_query import (
    chat_with_warehouse_system,
    get_session_history
)

# =====================================================================
# Page Config
# =====================================================================
sys.path.append(os.path.abspath(os.getcwd()))

st.set_page_config(
    page_title="AI Warehouse Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================================
# Load Custom CSS
# =====================================================================
def load_css():
    css_file = Path(__file__).parent / "styles.css"
    if css_file.exists():
        with open(css_file, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ ไม่พบไฟล์ CSS ที่: {css_file}")

load_css()

# =====================================================================
# Session init (ปรับปรุงให้ Clean)
# =====================================================================
if "user_id" not in st.session_state:
    st.session_state.user_id = "user_session"

if "chat_history" not in st.session_state:
    db = get_session_history(st.session_state.user_id)
    st.session_state.chat_history = db.messages

# =====================================================================
# Sidebar (Nav เข้มตามสไตล์ที่ต้องการ)
# =====================================================================
with st.sidebar:
    st.markdown("### การตั้งค่า")
    if st.button("เริ่มแชทใหม่", use_container_width=True):
        db = get_session_history(st.session_state.user_id)
        db.clear()
        st.session_state.chat_history = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### เกี่ยวกับระบบ")
    st.markdown("ระบบ AI ช่วยค้นหาข้อมูลคลังสินค้า")

# =====================================================================
# Header Area
# =====================================================================
st.markdown("<h1 style='text-align: center;'>AI Warehouse Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>ผู้ช่วยค้นหาข้อมูลคลังสินค้าอัจฉริยะ</p>", unsafe_allow_html=True)
st.markdown("---")

# =====================================================================
# 1. Chat Area (Container สำหรับข้อความแชท)
# =====================================================================
# ใช้ Container เพื่อล็อกพื้นที่ส่วนบนให้ข้อความแชทไหลอยู่ข้างใน
chat_container = st.container()

with chat_container:
    if not st.session_state.chat_history:
        st.info("👋 ยินดีต้อนรับ! พิมพ์คำถามเกี่ยวกับคลังสินค้าเพื่อเริ่มต้น")
    
    for msg in st.session_state.chat_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        avatar = "👤" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg.content)

# =====================================================================
# 2. Input Zone (วางไว้ท้ายสุด เพื่อให้คงอยู่ที่ด้านล่างจอ)
# =====================================================================
# การวาง chat_input ไว้โดดๆ นอก Container จะทำให้ Streamlit ตรึงมันไว้ที่ขอบล่างเสมอ
prompt = st.chat_input("💬 พิมพ์คำถามของคุณที่นี่...")

if prompt:
    # แสดงข้อความที่ผู้ใช้พิมพ์ใน Container ทันที
    with chat_container:
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # เริ่มประมวลผลคำตอบจาก AI
        with st.chat_message("assistant", avatar="🤖"):
            response_placeholder = st.empty()
            full_text = ""
            
            try:
                # เรียกใช้ Generator แบบ Streaming
                for chunk in chat_with_warehouse_system(st.session_state.user_id, prompt):
                    full_text += chunk
                    response_placeholder.markdown(full_text + " ▌")
                
                response_placeholder.markdown(full_text)
                
                # อัปเดต History ใน Session State ทันทีเพื่อให้แสดงผลต่อเนื่อง
                st.session_state.chat_history.append(HumanMessage(content=prompt))
                st.session_state.chat_history.append(AIMessage(content=full_text))

            except Exception as e:
                st.error(f"เกิดข้อผิดพลาด: {str(e)}")

# =====================================================================
# Footer
# =====================================================================
st.markdown("---")
st.markdown("<p style='text-align: center; opacity: 0.5;'><small>Powered by LangChain + Ollama</small></p>", unsafe_allow_html=True)