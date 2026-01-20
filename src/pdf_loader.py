from typing import List
# to use type hints
from pypdf import PdfReader
# to open and read PDF file to extract texts
import io
# for in-memory file handling

def load_pdf_pages(pdf_path:str) -> List[str]:
    # pdf_path contains the path to the pdf "Lorebook"
    # The function will return a list of string
    # Each element of the list represents the texts of each page

    reader = PdfReader(pdf_path)
    # creating a reader object that opens the PDF file
    # reader.pages will have a list-like object the contains the page objects

    pages_text : List[str] = []
    # Initlializing an empty list
    # It will be used to store text of each page

    for page in reader.pages:
        text = page.extract_text() or ""
        # Here, we are calling extract_text() on page object to read text content
        # Sometimes, they don't have extractable text, so it returns None
        # So, if it returns None, text becomes an empty string
        text = text.strip()
        # strip() removes leading and trailing spaces, newline characters at start and end
        pages_text.append(text)
        # Now, add cleaned text to the page_text list

    return pages_text


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Extract text from PDF bytes (in-memory, no disk I/O).
    
    Takes raw PDF file bytes and extracts all text content.
    Joins all pages into a single continuous string.
    
    Args:
        pdf_bytes: Raw PDF file content as bytes
    
    Returns:
        Extracted text as single string (all pages joined with newlines)
    
    Raises:
        ValueError: If PDF extraction fails or no text is found
    """
    try:
        # Create in-memory file object from bytes
        pdf_file = io.BytesIO(pdf_bytes)
        reader = PdfReader(pdf_file)
        
        if not reader.pages:
            raise ValueError("PDF has no pages")
        
        pages_text: List[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            text = text.strip()
            if text:  # Only add non-empty pages
                pages_text.append(text)
        
        if not pages_text:
            raise ValueError("No text could be extracted from PDF. Ensure it's a text-based PDF (not image-only).")
        
        # Join all pages with double newlines to preserve page breaks
        full_text = "\n\n".join(pages_text)
        return full_text
    
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")

if __name__ == "__main__":
    # will run this block only if you run this file directly
    import os
    # used here for path handling

    pdf_path = os.path.join("data", "Arpit_Negi_Resume.pdf")
    # Builds the file path
    pages = load_pdf_pages(pdf_path)
    # will return the list of page text, stored in pages
    print(f"Loaded {len(pages)} pages.")
    # will return how many pages were read
    print("First 500 characters of page 1: \n")
    if pages:
        # Check if the list is not empty
        print(pages[0][:500])

