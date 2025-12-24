from pathlib import Path
import streamlit as st
from datetime import datetime
from src.rag_query import get_session_history, chat_with_warehouse_system
from langchain_core.messages import HumanMessage, AIMessage
import time
import io
from PIL import Image

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

st.set_page_config(
    page_title="AI Warehouse Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

SESSION_ID_DEFAULT = "user_session"
QUICK_QUESTIONS = [
    "มีอะไหล่อะไรบ้างในคลัง?",
    "ช่วยหา Serial Number ของสินค้า",
    "สินค้าที่เลิกใช้งานมีอะไรบ้าง?",
    "แสดงข้อมูลสินค้าทั้งหมด"
]

# ============================================================================
# STYLING & SESSION
# ============================================================================

def load_custom_css():
    """โหลด CSS จากไฟล์ภายนอก"""
    css_path = Path(__file__).parent / "styles.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def initialize_session_state():
    """เริ่มต้น Session State"""
    defaults = {
        'user_id': SESSION_ID_DEFAULT,
        'is_processing': False,
        'stop_requested': False,
        'uploader_key': 0,
        'pending_data': None,
        'preview_image': None,
        'last_response_time': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # ดึงประวัติจากฐานข้อมูล
    history_db = get_session_history(st.session_state.user_id)
    st.session_state.chat_history = history_db.messages
    st.session_state.total_messages = len(history_db.messages)

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def clear_chat_history():
    """ล้างประวัติการสนทนาทั้งหมด"""
    history_db = get_session_history(st.session_state.user_id)
    history_db.clear()
    st.session_state.chat_history = []
    st.session_state.total_messages = 0
    st.session_state.preview_image = None
    st.success("✅ ล้างประวัติการสนทนาเรียบร้อยแล้ว")
    time.sleep(0.5)
    st.rerun()

def compress_image(image_bytes: bytes, max_size_kb: int = 500) -> bytes:
    """บีบอัดรูปภาพให้มีขนาดเหมาะสม เพื่อลดเวลาประมวลผล"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        
        # ปรับขนาดถ้ารูปใหญ่เกินไป
        max_dimension = 1024
        if max(img.size) > max_dimension:
            ratio = max_dimension / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # บันทึกในหน่วยความจำ
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=85, optimize=True)
        compressed = output.getvalue()
        
        # ตรวจสอบขนาด
        size_kb = len(compressed) / 1024
        print(f"📸 Image compressed: {len(image_bytes)/1024:.1f}KB → {size_kb:.1f}KB")
        
        return compressed
    except Exception as e:
        print(f"⚠️ Image compression failed: {e}")
        return image_bytes

# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_header():
    """แสดง Header พร้อม Subtitle"""
    st.markdown('''
        <div class="main-header">🤖 AI Warehouse Assistant</div>
        <div class="sub-header">ผู้ช่วยค้นหาข้อมูลคลังสินค้าอัจฉริยะ</div>
    ''', unsafe_allow_html=True)

def render_info_box():
    """แสดง Info Box พร้อมคำแนะนำ"""
    st.markdown("""
    <div class="info-box">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 1rem;">
            <div style="font-size: 2rem;">💡</div>
            <b style="font-size: 1.3rem; margin: 0;">วิธีใช้งาน</b>
        </div>
        <div style="padding-left: 1rem; line-height: 1.9; color: #e3e3e3;">
            • ค้นหาข้อมูลจาก <strong style="color: #667eea;">Serial Number</strong><br>
            • ตรวจสอบ <strong style="color: #667eea;">Model / รุ่นสินค้า</strong><br>
            • ตรวจสอบ <strong style="color: #667eea;">ตำแหน่งในคลัง</strong><br>
            • <strong style="color: #f59e0b;">📸 อัปโหลดรูปภาพ</strong> เพื่อหาข้อมูลจากรหัสในรูป
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Sidebar สำหรับจัดการแชทและสถิติ"""
    with st.sidebar:
        st.markdown("### 🎛️ เมนู")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 เริ่มใหม่", use_container_width=True, type="primary"):
                clear_chat_history()
        
        with col2:
            if st.button("🗑️ ล้างแชท", use_container_width=True):
                clear_chat_history()
        
        st.markdown("---")
        
        # สถิติการใช้งาน
        history_db = get_session_history(st.session_state.user_id)
        total_msgs = len(history_db.messages)
        user_msgs = len([m for m in history_db.messages if isinstance(m, HumanMessage)])
        
        st.markdown("### 📊 สถิติการใช้งาน")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("ข้อความทั้งหมด", total_msgs)
        with col2:
            st.metric("คำถามของคุณ", user_msgs)
        
        
        st.markdown("---")
        
        # ดาวน์โหลดประวัติ
        if history_db.messages:
            st.markdown("### 💾 บันทึกข้อมูล")
            
            transcript = f"AI Warehouse Assistant - Chat History\n"
            transcript += f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            transcript += f"Total Messages: {total_msgs}\n"
            transcript += "="*60 + "\n\n"
            
            for i, msg in enumerate(history_db.messages, 1):
                role = "👤 User" if isinstance(msg, HumanMessage) else "🤖 Assistant"
                timestamp = datetime.now().strftime('%H:%M:%S')
                transcript += f"[{timestamp}] {role} (Message {i}):\n{msg.content}\n\n"
            
            st.download_button(
                label="📥 ดาวน์โหลดประวัติ",
                data=transcript,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        st.markdown("---")
        
        # ข้อมูลระบบ
        st.markdown("### ℹ️ เกี่ยวกับระบบ")
        st.markdown("""
        <div style='font-size: 0.85rem; color: #b0b8c1;'>
        <strong>LLM:</strong> Gemma 3 (Ollama)<br>
        <strong>Vision:</strong> LLaVA / Gemma 3<br>
        <strong>Embedding:</strong> mxbai-embed-large<br>
        <strong>Vector DB:</strong> PostgreSQL + pgvector<br>
        <strong>Version:</strong> 2.1.0
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # คำแนะนำ
        with st.expander("💡 เคล็ดลับการใช้งาน"):
            st.markdown("""
            **สำหรับการค้นหาที่ดีที่สุด:**
            
            1. 📝 ระบุ Serial Number หรือ Model ที่ชัดเจน
            2. 📸 ถ่ายรูปป้ายรหัสให้ชัดเจน มีแสงสว่างพอ
            3. 🔍 ถ้าไม่เจอ ลองค้นด้วยคำอื่น
            4. 💬 ถามทีละเรื่อง จะได้ผลลัพธ์ดีกว่า
            """)

def render_chat_interface():
    """แสดงประวัติการแชท"""
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user", avatar="👤"):
                st.markdown(message.content)
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(message.content)

def render_welcome_screen():
    """แสดงหน้าต้อนรับพร้อมปุ่มคำถามแนะนำ"""
    if len(st.session_state.chat_history) > 0:
        return
    
    render_info_box()
    
    st.markdown('<div style="height: 2rem;"></div>', unsafe_allow_html=True)
    st.markdown('### 💬 คำถามที่ถามบ่อย')
    st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    for i, question in enumerate(QUICK_QUESTIONS):
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            if st.button(
                question, 
                use_container_width=True, 
                key=f"quick_{i}",
                help=f"คลิกเพื่อถาม: {question}"
            ):
                st.session_state.pending_data = {
                    'prompt': question,
                    'image': None,
                    'original_prompt': question
                }
                st.session_state.is_processing = True
                st.rerun()

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    load_custom_css()
    initialize_session_state()
    render_sidebar()
    render_header()

    # Container สำหรับแชท
    chat_holder = st.container()
    with chat_holder:
        if st.session_state.chat_history:
            render_chat_interface()
        else:
            render_welcome_screen()

    # Preview Zone (แสดงรูปที่เลือก)
    preview_placeholder = st.empty()

    # Chat Input Bar
    col_plus, col_input, col_stop = st.columns([1, 10, 1], gap="small")
    
    with col_plus:
        with st.popover("➕", help="อัปโหลดรูปภาพ"):
            uploaded_file = st.file_uploader(
                "เลือกรูปภาพ (รองรับ JPG, PNG)", 
                type=['png', 'jpg', 'jpeg'],
                key=f"uploader_{st.session_state.uploader_key}", 
                label_visibility="collapsed",
                help="อัปโหลดรูปภาพที่มีรหัสสินค้า"
            )
    
    with col_input:
        input_label = "⌛ กำลังประมวลผล..." if st.session_state.is_processing else "💭 พิมพ์คำถามหรือรหัสสินค้า... (หรืออัปโหลดรูป)"
        prompt = st.chat_input(input_label, disabled=st.session_state.is_processing)
    
    with col_stop:
        if st.session_state.is_processing:
            if st.button("⏹", help="หยุดการตอบกลับ", use_container_width=True):
                st.session_state.stop_requested = True
                st.session_state.is_processing = False
                st.warning("⚠️ หยุดการประมวลผล")
                time.sleep(0.5)
                st.rerun()

    # แสดง Preview รูปภาพ
    image_bytes = None
    if uploaded_file and not st.session_state.is_processing:
        image_bytes = uploaded_file.getvalue()
        
        # บีบอัดรูปเพื่อประมวลผลเร็วขึ้น
        image_bytes = compress_image(image_bytes)
        
        with preview_placeholder.container(border=True):
            col_img, col_info = st.columns([1, 5])
            with col_img:
                st.image(image_bytes, width=80)
            with col_info:
                st.write(f"🖼️ **พร้อมส่ง:** {uploaded_file.name}")
                st.caption(f"ขนาด: {len(image_bytes)/1024:.1f} KB")
                if st.button("❌ ยกเลิก", key="cancel_img"):
                    st.session_state.uploader_key += 1
                    st.rerun()

    # ส่งข้อมูล (เมื่อกด Enter)
    if prompt is not None:
        final_prompt = prompt.strip() if prompt.strip() else "ช่วยตรวจสอบข้อมูลจากรูปภาพนี้"
        
        st.session_state.pending_data = {
            'prompt': final_prompt,
            'image': image_bytes,
            'original_prompt': prompt.strip()
        }
        
        preview_placeholder.empty()
        st.session_state.is_processing = True
        st.session_state.uploader_key += 1
        st.rerun()

    # ประมวลผล
    if st.session_state.is_processing and st.session_state.pending_data:
        data = st.session_state.pending_data
        st.session_state.pending_data = None
        
        # แสดงข้อความผู้ใช้
        with chat_holder:
            with st.chat_message("user", avatar="👤"):
                if data['original_prompt']:
                    st.markdown(data['original_prompt'])
                if data['image']:
                    st.image(data['image'], width=300, caption="รูปภาพที่อัปโหลด")
        
        # AI ตอบ (Streaming)
        with chat_holder:
            with st.chat_message("assistant", avatar="🤖"):
                response_placeholder = st.empty()
                full_response = ""
                
                start_time = time.time()
                
                try:
                    # แสดง Loading
                    with st.spinner("🔄 กำลังวิเคราะห์..."):
                        response_gen = chat_with_warehouse_system(
                            st.session_state.user_id,
                            data['prompt'],
                            data['image']
                        )
                    
                    # Stream Response
                    has_response = False
                    for chunk in response_gen:
                        has_response = True
                        
                        if st.session_state.stop_requested:
                            full_response += "\n\n*[การตอบกลับถูกหยุดโดยผู้ใช้]*"
                            st.session_state.stop_requested = False
                            break
                        
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
                    
                    # แสดงผลสุดท้าย
                    if not has_response or not full_response.strip():
                        full_response = "⚠️ ไม่สามารถประมวลผลได้ กรุณาลองใหม่"
                    
                    response_placeholder.markdown(full_response)
                    
                    # บันทึกเวลา
                    elapsed = time.time() - start_time
                    st.session_state.last_response_time = elapsed
                    
                    # อัปเดตประวัติ
                    history_db = get_session_history(st.session_state.user_id)
                    st.session_state.chat_history = history_db.messages
                    
                    # แสดงสถานะสำเร็จ
                    st.caption(f"✅ ตอบเสร็จใน {elapsed:.2f} วินาที")
                    
                except Exception as e:
                    error_msg = f"""
                    ⚠️ **เกิดข้อผิดพลาด:**
                    
                    ```
                    {str(e)}
                    ```
                    
                    **💡 คำแนะนำ:**
                    - ตรวจสอบว่า Ollama กำลังทำงานอยู่ (`ollama list`)
                    - ลอง restart Ollama: `ollama serve`
                    - ตรวจสอบ Vision Model (ถ้าส่งรูป): `ollama pull llava`
                    - ดูรายละเอียดใน Terminal
                    """
                    
                    response_placeholder.markdown(error_msg)
                    print(f"❌ Streamlit Error: {e}")
                    import traceback
                    traceback.print_exc()
                
                finally:
                    st.session_state.is_processing = False
                    time.sleep(0.5)
                    st.rerun()

# ============================================================================
if __name__ == "__main__":
    main()