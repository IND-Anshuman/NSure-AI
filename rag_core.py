# utils.py
"""
PDF processing utilities
"""
import requests
import fitz  # PyMuPDF

def get_pdf_text_from_url(url: str) -> str:
    """Download PDF from URL and extract all text"""
    
    try:
        print(f"📥 Downloading PDF from: {url}")
        
        # Download the PDF
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        print("📖 Extracting text...")
        
        # Open PDF from bytes
        pdf_doc = fitz.open(stream=response.content, filetype="pdf")
        
        # Extract text from all pages
        full_text = ""
        for page_num in range(pdf_doc.page_count):
            page = pdf_doc[page_num]
            text = page.get_text()
            full_text += text + "\n"
        
        pdf_doc.close()
        
        if not full_text.strip():
            print("⚠️ No text found in PDF")
            return ""
        
        print(f"✅ Extracted {len(full_text)} characters from {pdf_doc.page_count} pages")
        return full_text
        
    except requests.RequestException as e:
        print(f"❌ Download failed: {e}")
        return ""
    except Exception as e:
        print(f"❌ PDF processing failed: {e}")
        return ""