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

EXCEL_FILE = "data/data_inventory.xlsx"

# ⚠️ แก้ชื่อ sheet ให้ตรงกับไฟล์ Excel ของคุณ
TARGET_SHEETS = [
    ("Spare", "อะไหล่สำรอง"),
    ("Obsolete", "สินค้าเลิกใช้งาน/เสื่อมสภาพ"),
]

# หรือถ้าชื่อ sheet ต่างกัน ให้เปลี่ยนเป็น:
# TARGET_SHEETS = [
#     ("ชื่อ Sheet 1 จริง", "Label ที่ต้องการ 1"),
#     ("ชื่อ Sheet 2 จริง", "Label ที่ต้องการ 2"),
# ]

# ============================================================================
# TEXT CLEANING
# ============================================================================
USELESS_VALUES = {
    "nan", "none", "null", "n/a", "", "-", "ไม่มี", "ไม่มีข้อมูล"
}

def clean_text(value: Optional[str]) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    
    text = str(value).strip()
    if text.lower() in USELESS_VALUES or text == "-":
        return None
    
    return text

# ============================================================================
# IMPROVED CONTENT BUILDER (สร้าง Context ที่ดีกว่า)
# ============================================================================
def build_rich_content(row: Dict, label: str) -> str:
    """สร้าง content ที่ rich และค้นหาง่ายขึ้น"""
    
    parts = []
    parts.append(f"ประเภท: {label}")
    parts.append("")
    
    # ดึงข้อมูลสำคัญ
    model = clean_text(row.get('Model'))
    model_no = clean_text(row.get('Model No.'))
    model_name = clean_text(row.get('Model Name'))
    serial = clean_text(row.get('Serial'))
    location = clean_text(row.get('Locations'))
    status = clean_text(row.get('Status'))
    asset_no = clean_text(row.get('Asset No'))
    lifetime = clean_text(row.get('Lifetime'))
    purchaser = clean_text(row.get('Purchaser'))
    order_number = clean_text(row.get('Order Number'))
    
    # สร้างหัวข้อหลัก - ใช้คำที่ค้นหาได้หลายแบบ
    if model:
        parts.append(f"สินค้า: {model}")
        parts.append(f"Model: {model}")
        parts.append(f"รุ่น: {model}")
        # เพิ่มการค้นหาแบบไม่มีช่องว่าง
        model_compact = model.replace(" ", "")
        parts.append(f"รหัสสินค้า: {model_compact}")
    
    if model_no:
        parts.append(f"Model Number: {model_no}")
        parts.append(f"หมายเลขรุ่น: {model_no}")
        parts.append(f"รหัสรุ่น: {model_no}")
    
    if model_name:
        parts.append(f"ชื่อรุ่น: {model_name}")
    
    parts.append("")
    parts.append("## ข้อมูลระบุตัวตน")
    
    if serial:
        parts.append(f"Serial Number: {serial}")
        parts.append(f"S/N: {serial}")
        parts.append(f"หมายเลขซีเรียล: {serial}")
        parts.append(f"ซีเรียลนัมเบอร์: {serial}")
    
    if asset_no:
        parts.append(f"Asset Number: {asset_no}")
        parts.append(f"Asset No: {asset_no}")
        parts.append(f"หมายเลขทรัพย์สิน: {asset_no}")
        parts.append(f"รหัสทรัพย์สิน: {asset_no}")
    
    parts.append("")
    parts.append("## ตำแหน่งและสถานะ")
    
    if location:
        parts.append(f"ตำแหน่ง: {location}")
        parts.append(f"Location: {location}")
        parts.append(f"อยู่ที่: {location}")
        parts.append(f"สถานที่: {location}")
    
    if status:
        parts.append(f"สถานะ: {status}")
        parts.append(f"Status: {status}")
    
    if lifetime:
        parts.append(f"อายุการใช้งาน: {lifetime}")
        parts.append(f"Lifetime: {lifetime}")
    
    parts.append("")
    parts.append("## ข้อมูลการจัดซื้อ")
    
    if purchaser:
        parts.append(f"ผู้จัดซื้อ: {purchaser}")
        parts.append(f"Purchaser: {purchaser}")
    
    if order_number:
        parts.append(f"เลขที่ใบสั่งซื้อ: {order_number}")
        parts.append(f"Order Number: {order_number}")
    
    # เพิ่มข้อมูลอื่นๆ ที่เหลือ
    parts.append("")
    parts.append("## รายละเอียดเพิ่มเติม")
    
    important_cols = ['Model', 'Model No.', 'Model Name', 'Serial', 'Status', 
                      'Lifetime', 'Purchaser', 'Order Number', 'Asset No', 'Locations']
    
    for col, val in row.items():
        if col not in important_cols:
            clean_val = clean_text(val)
            if clean_val:
                parts.append(f"{col}: {clean_val}")
    
    # เพิ่มส่วนสรุปสำหรับการค้นหา
    parts.append("")
    parts.append("## Context Hint")
    search_terms = []
    if model:
        search_terms.append(model)
        search_terms.append(model.replace(" ", ""))
    if serial:
        search_terms.append(f"Serial {serial}")
    if asset_no:
        search_terms.append(f"Asset {asset_no}")
    if location:
        search_terms.append(f"ที่ {location}")
    
    parts.append("คำค้นหาที่เกี่ยวข้อง: " + " | ".join(search_terms))
    
    return "\n".join(parts)

# ============================================================================
# EXCEL → DOCUMENT GENERATOR
# ============================================================================
def load_documents_from_sheet(
    file_path: str,
    sheet_name: str,
    label: str,
) -> List[Document]:
    
    print(f"  กำลังโหลด sheet: {sheet_name}")
    
    df = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        dtype=str,
    ).dropna(how="all")
    
    # ลบแถวที่ซ้ำกัน
    df = df.drop_duplicates()
    
    # ทำความสะอาดชื่อคอลัมน์
    df.columns = [str(c).strip() for c in df.columns]
    
    print(f"  พบข้อมูล {len(df)} แถว")
    
    documents: List[Document] = []
    
    for idx, row in df.iterrows():
        data = row.to_dict()
        content = build_rich_content(data, label)
        
        # ต้องมีเนื้อหาอย่างน้อย 30 ตัวอักษร
        if len(content) < 30:
            continue
        
        # สร้าง metadata ที่มีประโยชน์
        metadata = {
            "sheet": sheet_name,
            "category": label,
            "row": int(idx),
            "model": clean_text(data.get('Model', '')),
            "serial": clean_text(data.get('Serial', '')),
            "location": clean_text(data.get('Locations', '')),
        }
        
        # ลบ None ออกจาก metadata
        metadata = {k: v for k, v in metadata.items() if v is not None}
        
        documents.append(
            Document(
                page_content=content,
                metadata=metadata,
            )
        )
    
    print(f"  สร้างได้ {len(documents)} documents")
    
    del df
    gc.collect()
    
    return documents

# ============================================================================
# INGESTION PIPELINE
# ============================================================================
def ingest_to_pgvector():
    
    print("="*60)
    print("เริ่มต้น Ingestion Process")
    print("="*60)
    
    print("\n1. กำลังโหลดข้อมูลจาก Excel...")
    all_docs: List[Document] = []
    
    for sheet, label in TARGET_SHEETS:
        docs = load_documents_from_sheet(EXCEL_FILE, sheet, label)
        all_docs.extend(docs)
    
    if not all_docs:
        print("ไม่พบข้อมูลที่ valid")
        return
    
    print(f"\n✓ รวมทั้งหมด: {len(all_docs)} documents")
    
    # แสดงตัวอย่าง document แรก
    print("\n2. ตัวอย่างเนื้อหา document แรก:")
    print("-"*60)
    print(all_docs[0].page_content[:500])
    print("-"*60)
    
    # ---- Split documents (เปลี่ยนเป็น chunk ขนาดใหญ่ขึ้น)
    print("\n3. กำลัง split documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,       # เพิ่มเป็น 1200 เพื่อเก็บข้อมูลทั้งหมดของแต่ละรายการ
        chunk_overlap=200,     # overlap มากขึ้นเพื่อไม่ตัดข้อมูลสำคัญ
        separators=[
    "\n\n## ",
    "\n\n",
    "\n",
    ". ",
    " ",
],

    )
    
    chunks = splitter.split_documents(all_docs)
    print(f"✓ สร้าง {len(chunks)} chunks")
    
    # แสดงตัวอย่าง chunk
    print("\nตัวอย่าง chunk แรก:")
    print("-"*60)
    print(chunks[0].page_content[:400])
    print("-"*60)
    
    del all_docs
    gc.collect()
    
    # ---- Embedding + Store
    print("\n4. กำลังสร้าง embeddings และเขียนลง PGVector...")
    
    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    
    print(f"   Collection: {COLLECTION}")
    print("   กำลังลบ collection เก่า (ถ้ามี)...")
    
    PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION,
        connection_string=DB_URL,
        pre_delete_collection=True,  # ลบของเก่าก่อน
    )
    
    print("\n" + "="*60)
    print("✓ Ingestion สำเร็จ!")
    print("="*60)
    print(f"Collection: {COLLECTION}")
    print(f"จำนวน chunks: {len(chunks)}")
    print(f"ขนาด chunk: 800 characters")
    print("="*60)

# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    ingest_to_pgvector()