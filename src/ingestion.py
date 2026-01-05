import os
import gc
import pandas as pd
from typing import List, Dict, Optional
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ============================================================================
# Configuration
# ============================================================================
DB_URL = f"postgresql+psycopg2://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DATABASE')}"
COLLECTION = os.getenv("COLLECTION_NAME")
EMBED_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"

# ไฟล์ Excel จริงของคุณ
INVENTORY_FILE = "data/data_inventory.xlsx"

# Sheets ที่ต้องการ (ตรงกับไฟล์ของคุณ)
TARGET_SHEETS = [
    ("Spare", "อะไหล่/อุปกรณ์สำรอง"),
    ("Obsolete", "อุปกรณ์เลิกใช้งาน"),
]

# ============================================================================
# TEXT CLEANING
# ============================================================================
USELESS_VALUES = {
    "nan", "none", "null", "n/a", "", "-", "ไม่มี", "ไม่มีข้อมูล", "N/A"
}

def clean_text(value: Optional[str]) -> Optional[str]:
    """ทำความสะอาดข้อความ"""
    if value is None or pd.isna(value):
        return None
    
    text = str(value).strip()
    if text.lower() in USELESS_VALUES or text == "-":
        return None
    
    return text

# ============================================================================
# DEVICE CATEGORY DETECTION
# ============================================================================
def detect_device_category(model: str) -> str:
    """ตรวจจับประเภทอุปกรณ์จาก Model"""
    if not model:
        return "อุปกรณ์ทั่วไป"
    
    model_lower = model.lower()
    
    if "thinkpad" in model_lower or "laptop" in model_lower or "elitebook" in model_lower:
        return "Laptop/Notebook"
    elif "thinkcentre" in model_lower or "optiplex" in model_lower or "prodesk" in model_lower:
        return "Desktop Computer"
    elif "thinkstation" in model_lower or "workstation" in model_lower:
        return "Workstation"
    elif "switch" in model_lower:
        return "Network Switch"
    elif "router" in model_lower:
        return "Router"
    elif "access point" in model_lower or "wifi" in model_lower or "wireless" in model_lower:
        return "Access Point/WiFi"
    elif "neverstop" in model_lower or "printer" in model_lower:
        return "Printer"
    elif "mac mini" in model_lower or "mac" in model_lower:
        return "Mac Computer"
    elif "lenovo" in model_lower and ("v510z" in model_lower or "aio" in model_lower):
        return "All-in-One PC"
    elif "air" in model_lower and "4g" in model_lower:
        return "4G Router/Modem"
    else:
        return "อุปกรณ์ IT อื่นๆ"

# ============================================================================
# IMPROVED CONTENT BUILDER
# ============================================================================
def build_inventory_content(row: Dict, sheet_label: str) -> str:
    """สร้าง content ที่ rich สำหรับการค้นหา"""
    
    parts = []
    
    # Header
    parts.append(f"ประเภทข้อมูล: IT Asset Inventory")
    parts.append(f"หมวดหมู่: {sheet_label}")
    parts.append("")
    
    # ดึงข้อมูลหลัก
    model = clean_text(row.get('Model'))
    model_no = clean_text(row.get('Model No.'))
    model_name = clean_text(row.get('Model Name'))
    serial = clean_text(row.get('Serial'))
    status = clean_text(row.get('Status'))
    lifetime = clean_text(row.get('Lifetime'))
    purchased = clean_text(row.get('Purchased'))
    order_num = clean_text(row.get('Order Number'))
    asset_no = clean_text(row.get('Asset No'))
    locations = clean_text(row.get('Locations'))
    
    # ตรวจจับประเภทอุปกรณ์
    device_category = detect_device_category(model) if model else "ไม่ระบุ"
    parts.append(f"ประเภทอุปกรณ์: {device_category}")
    parts.append("")
    
    # ข้อมูลสินค้า/รุ่น (สร้างหลายรูปแบบเพื่อการค้นหา)
    parts.append("## ข้อมูลรุ่นและสินค้า")
    
    if model:
        parts.append(f"รุ่น: {model}")
        parts.append(f"Model: {model}")
        parts.append(f"สินค้า: {model}")
        # เพิ่มการค้นหาแบบไม่มีช่องว่าง
        model_compact = model.replace(" ", "")
        parts.append(f"รหัส: {model_compact}")
    
    if model_no:
        parts.append(f"Model Number: {model_no}")
        parts.append(f"หมายเลขรุ่น: {model_no}")
        parts.append(f"รหัสรุ่น: {model_no}")
    
    if model_name:
        parts.append(f"Model Name: {model_name}")
        parts.append(f"ชื่อรุ่น: {model_name}")
    
    parts.append("")
    parts.append("## ข้อมูลระบุตัวตน")
    
    if serial:
        parts.append(f"Serial Number: {serial}")
        parts.append(f"S/N: {serial}")
        parts.append(f"Serial: {serial}")
        parts.append(f"หมายเลขซีเรียล: {serial}")
        parts.append(f"ซีเรียล: {serial}")
    
    if asset_no:
        parts.append(f"Asset Number: {asset_no}")
        parts.append(f"Asset No: {asset_no}")
        parts.append(f"หมายเลขทรัพย์สิน: {asset_no}")
        parts.append(f"รหัสทรัพย์สิน: {asset_no}")
    
    parts.append("")
    parts.append("## สถานะและตำแหน่ง")
    
    if status:
        parts.append(f"สถานะ: {status}")
        parts.append(f"Status: {status}")
        
        # แปลสถานะให้เข้าใจง่าย
        if "spare" in status.lower():
            parts.append("เป็นอะไหล่สำรอง พร้อมใช้งาน available spare")
        elif "obsolete" in status.lower():
            if "deployable" in status.lower():
                parts.append("เลิกใช้แล้วแต่ยังใช้งานได้ obsolete but still working")
            elif "deployed" in status.lower():
                parts.append("เลิกใช้แล้วและถูกใช้งานอยู่ obsolete and in use")
    
    if locations:
        parts.append(f"ตำแหน่ง: {locations}")
        parts.append(f"Location: {locations}")
        parts.append(f"สถานที่: {locations}")
        parts.append(f"อยู่ที่: {locations}")
    
    parts.append("")
    parts.append("## ข้อมูลการจัดซื้อและอายุการใช้งาน")
    
    if lifetime:
        parts.append(f"อายุการใช้งาน: {lifetime}")
        parts.append(f"Lifetime: {lifetime}")
        parts.append(f"อายุ: {lifetime}")
    
    if purchased:
        parts.append(f"วันที่จัดซื้อ: {purchased}")
        parts.append(f"Purchased: {purchased}")
        parts.append(f"ซื้อเมื่อ: {purchased}")
    
    if order_num:
        parts.append(f"เลขที่ใบสั่งซื้อ: {order_num}")
        parts.append(f"Order Number: {order_num}")
        parts.append(f"PO: {order_num}")
    
    # ข้อมูลเพิ่มเติมอื่นๆ
    parts.append("")
    parts.append("## รายละเอียดเพิ่มเติม")
    
    important_cols = ['Model', 'Model No.', 'Model Name', 'Serial', 'Status', 
                      'Lifetime', 'Purchased', 'Order Number', 'Asset No', 'Locations']
    
    for col, val in row.items():
        if col not in important_cols:
            clean_val = clean_text(val)
            if clean_val:
                parts.append(f"{col}: {clean_val}")
    
    # Context Hint สำหรับการค้นหา
    parts.append("")
    parts.append("## คำค้นหาที่เกี่ยวข้อง")
    search_terms = []
    
    if model:
        search_terms.append(model)
        search_terms.append(model.replace(" ", ""))
    if serial:
        search_terms.append(f"Serial {serial}")
    if asset_no:
        search_terms.append(f"Asset {asset_no}")
    if locations:
        search_terms.append(f"ที่ {locations}")
    if device_category:
        search_terms.append(device_category)
    
    parts.append(" | ".join(search_terms))
    
    return "\n".join(parts)

# ============================================================================
# DOCUMENT LOADER
# ============================================================================
def load_inventory_documents(
    file_path: str,
    sheet_configs: List[tuple]
) -> List[Document]:
    """โหลดข้อมูล inventory จาก Excel"""
    
    all_docs = []
    
    for sheet_name, label in sheet_configs:
        print(f"\n  📄 กำลังโหลด sheet: {sheet_name}")
        
        try:
            df = pd.read_excel(
                file_path,
                sheet_name=sheet_name,
                dtype=str,
            ).dropna(how="all")
            
            # ลบแถวซ้ำ
            df = df.drop_duplicates()
            
            # ทำความสะอาดชื่อคอลัมน์
            df.columns = [str(c).strip() for c in df.columns]
            
            print(f"     พบข้อมูล {len(df)} แถว")
            
            # สร้าง Documents
            doc_count = 0
            for idx, row in df.iterrows():
                data = row.to_dict()
                
                # ต้องมี Model เป็นอย่างน้อย
                if not clean_text(data.get('Model')):
                    continue
                
                content = build_inventory_content(data, label)
                
                # ต้องมีเนื้อหาอย่างน้อย 50 ตัวอักษร
                if len(content) < 50:
                    continue
                
                # สร้าง metadata
                metadata = {
                    "source": "inventory",
                    "sheet": sheet_name,
                    "category": label,
                    "row": int(idx),
                    "model": clean_text(data.get('Model')),
                    "serial": clean_text(data.get('Serial')),
                    "asset_no": clean_text(data.get('Asset No')),
                    "location": clean_text(data.get('Locations')),
                    "status": clean_text(data.get('Status')),
                    "device_type": detect_device_category(clean_text(data.get('Model', ''))),
                }
                
                # ลบค่า None
                metadata = {k: v for k, v in metadata.items() if v is not None}
                
                all_docs.append(
                    Document(
                        page_content=content,
                        metadata=metadata,
                    )
                )
                doc_count += 1
            
            print(f"     ✓ สร้างได้ {doc_count} documents")
            
            del df
            gc.collect()
            
        except Exception as e:
            print(f"     ❌ Error loading {sheet_name}: {e}")
            continue
    
    return all_docs

# ============================================================================
# MAIN INGESTION
# ============================================================================
def ingest_real_inventory():
    
    print("="*70)
    print(" IT SUPPORT KNOWLEDGE BASE - INGESTION PROCESS")
    print("="*70)
    
    # ตรวจสอบไฟล์
    if not os.path.exists(INVENTORY_FILE):
        print(f"\nไม่พบไฟล์: {INVENTORY_FILE}")
        print("กรุณาตรวจสอบว่าไฟล์อยู่ในโฟลเดอร์ data/")
        return
    
    print(f"\n กำลังโหลดข้อมูลจาก: {INVENTORY_FILE}")
    
    # โหลดข้อมูล
    all_docs = load_inventory_documents(INVENTORY_FILE, TARGET_SHEETS)
    
    if not all_docs:
        print("\n ไม่พบข้อมูลที่ valid")
        return
    
    print(f"\n รวมทั้งหมด: {len(all_docs)} documents")
    
    # แสดงสถิติ
    sheet_stats = {}
    for doc in all_docs:
        sheet = doc.metadata.get('sheet', 'Unknown')
        sheet_stats[sheet] = sheet_stats.get(sheet, 0) + 1
    
    print("\n สถิติตาม Sheet:")
    for sheet, count in sheet_stats.items():
        print(f"   - {sheet}: {count} documents")
    
    # แสดงตัวอย่าง
    print("\n ตัวอย่าง Document แรก:")
    print("-"*70)
    print(all_docs[0].page_content[:600])
    print("...")
    print("-"*70)
    
    # Split documents
    print("\n กำลัง split documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # เพิ่มขนาดเพื่อเก็บข้อมูลครบ
        chunk_overlap=250,
        separators=["\n\n## ", "\n\n", "\n", ". ", " "],
    )
    
    chunks = splitter.split_documents(all_docs)
    print(f" สร้าง {len(chunks)} chunks")
    
    # แสดงตัวอย่าง chunk
    print("\n ตัวอย่าง chunk แรก:")
    print("-"*70)
    print(chunks[0].page_content[:500])
    print("...")
    print("-"*70)
    
    # เก็บจำนวนก่อนลบ
    total_docs = len(all_docs)
    total_chunks = len(chunks)
    
    del all_docs
    gc.collect()
    
    # Embedding + Store
    print(f"\n กำลังสร้าง embeddings และเขียนลง PGVector...")
    print(f"   Collection: {COLLECTION}")
    print(f"   Embedding Model: {EMBED_MODEL}")
    
    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    
    try:
        PGVector.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION,
            connection_string=DB_URL,
            pre_delete_collection=True,  # ลบของเก่า
        )
        
        print("\n" + "="*70)
        print(" INGESTION สำเร็จ!")
        print("="*70)
        print(f" Collection: {COLLECTION}")
        print(f" จำนวน Documents: {total_docs}")
        print(f" จำนวน Chunks: {total_chunks}")
        print(f" Chunk Size: 1500 characters")
        print(f" Overlap: 250 characters")
        print("="*70)
    except Exception as e:
        print(f"\n Error during ingestion: {e}")
        
# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    ingest_real_inventory()