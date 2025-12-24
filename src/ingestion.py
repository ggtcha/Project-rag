import os
import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict
import re

load_dotenv()

# ============================================================================
# Configuration
# ============================================================================
DB_URL = f"postgresql+psycopg2://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DATABASE')}"
COLLECTION = os.getenv("COLLECTION_NAME")
EMBED_MODEL = "mxbai-embed-large"
OLLAMA_BASE_URL = "http://localhost:11434"

# ============================================================================
# Utility Functions
# ============================================================================
def clean_text(value):
    """ล้างข้อมูลขยะโดยยังรักษา Serial Number ไว้"""
    if pd.isna(value) or value is None:
        return None
    
    val = str(value).strip()
    useless = ['nan', 'none', 'null', 'n/a', '', 'ไม่มี', 'ไม่มีข้อมูล']
    junk_patterns = ['dtype: object', 'dtype: int64', 'Name:', 'Unnamed:']
    
    if val.lower() in useless:
        return None
    
    for pattern in junk_patterns:
        val = val.replace(pattern, '')
        
    return val.strip() or None

def extract_searchable_codes(row_dict: Dict) -> List[str]:
    """ดึงรหัสที่สำคัญออกมาเพื่อใช้ในการค้นหา"""
    codes = []
    priority_fields = ['serial', 's/n', 'sn', 'model', 'part', 'asset', 'code']
    
    for col, val in row_dict.items():
        col_lower = str(col).lower()
        
        # เช็คว่าเป็น field ที่เก็บรหัสหรือไม่
        if any(pf in col_lower for pf in priority_fields):
            clean_val = clean_text(val)
            if clean_val:
                # ดึงรหัสด้วย regex
                found_codes = re.findall(r'[A-Z0-9]+-[A-Z0-9-/]+|[A-Z]*\d{5,}', str(clean_val).upper())
                codes.extend(found_codes)
    
    return list(set(codes))  # ลบ duplicate

def create_content_body(row_dict: Dict, label: str) -> str:
    """จัดโครงสร้างเนื้อหาแบบมีลำดับความสำคัญ"""
    groups = {
        "Identification": [], 
        "Location": [],      
        "Responsibility": [], 
        "Status": [],        
        "Technical": [],
        "Others": []
    }
    
    keywords = {
        "Identification": ['serial', 's/n', 'sn', 'part', 'model', 'รหัส', 'หมายเลข', 'code', 'name', 'asset', 'item'],
        "Location": ['location', 'ตำแหน่ง', 'สถานที่', 'โซน', 'shelf', 'zone', 'room', 'building', 'อาคาร'],
        "Responsibility": ['responsible', 'owner', 'ผู้รับผิดชอบ', 'เจ้าของ', 'person', 'user', 'department'],
        "Status": ['status', 'สถานะ', 'condition', 'state', 'available'],
        "Technical": ['spec', 'description', 'รายละเอียด', 'คุณสมบัติ', 'brand', 'manufacturer']
    }

    for col, raw_val in row_dict.items():
        val = clean_text(raw_val)
        if not val: 
            continue
        
        line = f"{col}: {val}"
        found = False
        
        for group, keys in keywords.items():
            if any(k in str(col).lower() for k in keys):
                groups[group].append(line)
                found = True
                break
        
        if not found: 
            groups["Others"].append(line)

    # สร้างเนื้อหาตามลำดับความสำคัญ
    sections = [f"=== Category: {label} ==="]
    
    titles = {
        "Identification": "## รหัสและชื่อสินค้า", 
        "Location": "## ตำแหน่งและสถานที่", 
        "Responsibility": "## ผู้รับผิดชอบ", 
        "Status": "## สถานะ", 
        "Technical": "## รายละเอียดทางเทคนิค",
        "Others": "## ข้อมูลเพิ่มเติม"
    }
    
    for key, title in titles.items():
        if groups[key]:
            sections.extend([f"\n{title}", *groups[key]])
    
    # เพิ่มรหัสที่สำคัญไว้ท้ายสุด เพื่อเพิ่มโอกาสค้นเจอ
    searchable_codes = extract_searchable_codes(row_dict)
    if searchable_codes:
        sections.append(f"\n## Searchable Codes: {', '.join(searchable_codes)}")
    
    return "\n".join(sections)

def process_sheet(file: str, sheet: str, label: str) -> List[Document]:
    """ประมวลผล Sheet และแปลงเป็น Documents"""
    try:
        df = pd.read_excel(file, sheet_name=sheet)
        
        # ทำความสะอาดข้อมูล
        df = df.dropna(how='all').drop_duplicates()
        df.columns = [str(c).strip() for c in df.columns]
        
        print(f"📋 Processing sheet: {sheet}")
        print(f"   - Total rows: {len(df)}")
        print(f"   - Columns: {list(df.columns)}")
        
        docs = []
        skipped = 0
        
        for idx, row in df.iterrows():
            data = row.to_dict()
            content = create_content_body(data, label)
            
            # ตรวจสอบว่ามีเนื้อหาเพียงพอหรือไม่
            if len(content) < 30: 
                skipped += 1
                continue
            
            # สร้าง metadata ที่สะอาด
            meta = {}
            for k, v in data.items():
                clean_v = clean_text(v)
                if clean_v:
                    meta[k.lower().replace(' ', '_')] = clean_v
            
            meta.update({
                "sheet": sheet, 
                "category": label, 
                "row_index": int(idx)
            })
            
            docs.append(Document(page_content=content, metadata=meta))
        
        print(f"   ✅ Created: {len(docs)} documents")
        print(f"   ⚠️ Skipped: {skipped} rows (insufficient data)")
        
        return docs
        
    except Exception as e:
        print(f"❌ Error processing {sheet}: {e}")
        import traceback
        traceback.print_exc()
        return []

def verify_documents(docs: List[Document]):
    """ตรวจสอบคุณภาพของเอกสารก่อนนำเข้า"""
    print("\n🔍 Verifying Documents Quality...")
    
    if not docs:
        print("❌ No documents to verify!")
        return False
    
    # ตรวจสอบตัวอย่างเอกสาร
    sample = docs[0]
    print(f"\n📄 Sample Document:")
    print(f"Content preview: {sample.page_content[:200]}...")
    print(f"Metadata: {sample.metadata}")
    
    # ตรวจสอบรหัสที่สำคัญ
    codes_found = 0
    for doc in docs[:10]:  # ตรวจ 10 อันแรก
        if "Searchable Codes:" in doc.page_content:
            codes_found += 1
    
    print(f"\n📊 Statistics:")
    print(f"   - Total documents: {len(docs)}")
    print(f"   - Documents with codes: {codes_found}/10 (sample)")
    print(f"   - Average length: {sum(len(d.page_content) for d in docs) / len(docs):.0f} chars")
    
    return True

# ============================================================================
# Main Ingestion
# ============================================================================
def run_ingestion():
    print("="*60)
    print("🚀 Starting Data Ingestion Process")
    print("="*60)
    
    excel_file = "data/data_inventory.xlsx" 
    
    # ตรวจสอบไฟล์
    if not os.path.exists(excel_file):
        print(f"❌ File not found: {excel_file}")
        return
    
    target_sheets = [
        ("Spare", "อะไหล่สำรอง"), 
        ("Obsolete", "สินค้าเลิกใช้งาน/เสื่อมสภาพ")
    ]
    
    print(f"\n📂 Reading from: {excel_file}")
    print(f"📊 Target sheets: {[s[0] for s in target_sheets]}\n")
    
    # อ่านข้อมูล
    all_docs = []
    for sheet, label in target_sheets:
        docs = process_sheet(excel_file, sheet, label)
        all_docs.extend(docs)

    if not all_docs:
        print("\n❌ No valid documents found!")
        return

    print(f"\n{'='*60}")
    print(f"📦 Total Documents: {len(all_docs)}")
    print(f"{'='*60}\n")
    
    # ตรวจสอบคุณภาพ
    if not verify_documents(all_docs):
        print("❌ Document verification failed!")
        return
    
    # แบ่งเอกสาร
    print("\n✂️ Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,        # เพิ่มขนาด chunk
        chunk_overlap=150,     # เพิ่ม overlap
        separators=[
            "=== Category:",
            "\n## ",
            "\n\n",
            "\n",
            " "
        ],
        length_function=len,
    )
    
    chunks = splitter.split_documents(all_docs)
    print(f"✅ Created {len(chunks)} chunks")
    
    # แสดงตัวอย่าง chunk
    if chunks:
        print(f"\n📄 Sample Chunk:")
        print(f"{chunks[0].page_content[:300]}...")
    
    # บันทึกลงฐานข้อมูล
    try:
        print(f"\n{'='*60}")
        print("🗄️ Storing in Vector Database...")
        print(f"{'='*60}\n")
        
        embeds = OllamaEmbeddings(
            model=EMBED_MODEL, 
            base_url=OLLAMA_BASE_URL
        )
        
        print("⏳ This may take a few minutes...")
        
        PGVector.from_documents(
            embedding=embeds,
            documents=chunks,
            collection_name=COLLECTION,
            connection_string=DB_URL,
            pre_delete_collection=True,
            use_jsonb=True
        )
        
        print(f"\n{'='*60}")
        print("✅ INGESTION COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"📊 Collection: {COLLECTION}")
        print(f"📦 Total chunks: {len(chunks)}")
        print(f"🎯 Ready for queries!")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ DATABASE ERROR")
        print(f"{'='*60}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_ingestion()