import requests
import fitz

def get_pdf_text_from_url(url: str) -> str:
    print(f"📥 Downloading PDF from: {url}")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        pdf_bytes = response.content
        
        print("📖 Extracting text...")
        text_content = ""
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            print(f"Successfully opened PDF. It has {len(doc)} pages.")
            for page_num, page in enumerate(doc):
                text_content += page.get_text()
            print("Text extraction complete.")
        
        if not text_content.strip():
            print("⚠️ No text found in PDF")
            return ""
        
        print(f"✅ Extracted {len(text_content)} characters")
        return text_content
        
    except requests.RequestException as e:
        print(f"❌ Download failed: {e}")
        return ""
    except Exception as e:
        print(f"❌ PDF processing failed: {e}")
        return ""