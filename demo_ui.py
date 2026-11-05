import sys, types, time, runpy
from langchain_core.messages import HumanMessage, AIMessage

if "src.rag_query" not in sys.modules:
    class _History:
        def __init__(self): self.messages = []
        def add_user_message(self, t): self.messages.append(HumanMessage(content=t))
        def add_ai_message(self, t): self.messages.append(AIMessage(content=t))
        def clear(self): self.messages = []

    _store = {}
    def get_session_history(sid):
        return _store.setdefault(sid, _History())

    def chat_with_warehouse_system(sid, question, image=None):
        answer = (
            "🔍 พบข้อมูลตัวอย่าง 1 รายการ\n\n"
            "- **Model:** ThinkPad T14\n"
            "- **Serial:** XXXXXXXXXX\n"
            "- **สถานะ:** Spare\n"
            "- **ตำแหน่ง:** ศรีราชา\n"
        )
        out = ""
        for ch in answer:
            out += ch
            yield ch
            time.sleep(0.01)
        h = get_session_history(sid)
        h.add_user_message(question)
        h.add_ai_message(out)

    pkg = types.ModuleType("src"); pkg.__path__ = []
    mod = types.ModuleType("src.rag_query")
    mod.get_session_history = get_session_history
    mod.chat_with_warehouse_system = chat_with_warehouse_system
    sys.modules["src"] = pkg
    sys.modules["src.rag_query"] = mod

runpy.run_path("app.py", run_name="__main__")
