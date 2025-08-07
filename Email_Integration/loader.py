def extract_text_from_source(txt_path: str) -> str:
    with open(txt_path, 'r', encoding='utf-8') as txt:
        return txt.read()
print("✅ loader.py loaded successfully")
