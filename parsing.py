import os
from typing import Optional, Union
from io import BytesIO
import magic # You may need to install python-magic for this

# PDF parsing
from pdfminer.high_level import extract_text as extract_text_from_pdf

# DOCX parsing
import docx2txt

# Email parsing
import mailparser

# To detect file type from bytes, we'll use a library like python-magic
# You will need to install this: `pip install python-magic python-magic-bin` (for Windows)
# Or just `pip install python-magic` (for Linux/macOS, requires libmagic)

def detect_file_type(file_source: Union[str, bytes]) -> Optional[str]:
    """Detects the file type based on the content or extension."""
    if isinstance(file_source, str):
        # If it's a file path, guess from the extension
        ext = os.path.splitext(file_source)[1].lower()
        if ext == ".pdf":
            return "pdf"
        elif ext == ".docx":
            return "docx"
        elif ext in [".eml", ".msg"]:
            return "email"
        else:
            return None
    elif isinstance(file_source, bytes):
        # If it's raw bytes, use python-magic to detect the MIME type
        try:
            mime = magic.from_buffer(file_source, mime=True)
            if mime == "application/pdf":
                return "pdf"
            elif mime in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/zip"]:
                # docx files are technically zip files, so we might need a more specific check
                return "docx"
            elif mime in ["message/rfc822", "application/vnd.ms-outlook"]:
                return "email"
            else:
                return None
        except magic.MagicException:
            # Fallback for systems where magic is not configured correctly.
            # We can do a simple check on the first few bytes.
            if file_source.startswith(b"%PDF"):
                return "pdf"
            elif file_source.startswith(b"PK\x03\x04") and b"word/" in file_source:
                return "docx"
            # Detecting emails from bytes is less reliable, but possible
            return None
    return None


def parse_pdf(file_source: Union[str, bytes]) -> str:
    """Parses a PDF from a file path or bytes."""
    try:
        if isinstance(file_source, str):
            text = extract_text_from_pdf(file_source)
        elif isinstance(file_source, bytes):
            text = extract_text_from_pdf(BytesIO(file_source))
        else:
            raise TypeError("file_source must be a file path (str) or raw bytes (bytes).")
        return text
    except Exception as e:
        source_desc = "in-memory bytes" if isinstance(file_source, bytes) else file_source
        print(f"Error parsing PDF from {source_desc}: {e}")
        return ""


def parse_docx(file_source: Union[str, bytes]) -> str:
    """Parses a DOCX from a file path or bytes."""
    try:
        if isinstance(file_source, str):
            text = docx2txt.process(file_source)
        elif isinstance(file_source, bytes):
            # docx2txt.process can take a BytesIO object directly.
            # This requires a specific fork or newer version. A safer,
            # more universal approach is to write to a temp file, but
            # that defeats the purpose of in-memory processing.
            # We will use the in-memory approach which works for modern docx2txt
            text = docx2txt.process(BytesIO(file_source))
        else:
            raise TypeError("file_source must be a file path (str) or raw bytes (bytes).")
        return text
    except Exception as e:
        source_desc = "in-memory bytes" if isinstance(file_source, bytes) else file_source
        print(f"Error parsing DOCX from {source_desc}: {e}")
        return ""


def parse_email(file_source: Union[str, bytes]) -> str:
    """Parses an email from a file path or bytes."""
    try:
        if isinstance(file_source, str):
            mail = mailparser.parse_from_file(file_source)
        elif isinstance(file_source, bytes):
            mail = mailparser.parse_from_bytes(file_source)
        else:
            raise TypeError("file_source must be a file path (str) or raw bytes (bytes).")
        
        subject = mail.subject or ""
        body = mail.body or ""
        text = f"Subject: {subject}\n\n{body}"
        return text
    except Exception as e:
        source_desc = "in-memory bytes" if isinstance(file_source, bytes) else file_source
        print(f"Error parsing email from {source_desc}: {e}")
        return ""


def parse_document(file_source: Union[str, bytes]) -> Optional[str]:
    """
    Parses a document from a file path or raw bytes.
    The function intelligently determines the file type.
    """
    file_type = detect_file_type(file_source)
    
    if file_type == "pdf":
        return parse_pdf(file_source)
    elif file_type == "docx":
        return parse_docx(file_source)
    elif file_type == "email":
        return parse_email(file_source)
    else:
        print(f"Unsupported file type detected for source: {file_source}")
        return None


if __name__ == "__main__":
    # --- Example with file paths (for local testing) ---
    sample_files = ["dataset/d1.pdf"] # Replace with your test files
    with open("parsed.txt", "w", encoding="utf-8") as output_file:
        for f in sample_files:
            print(f"Parsing file from path: {f}")
            raw_text = parse_document(f)
            if raw_text:
                output_file.write(f"File: {f}\n")
                output_file.write(raw_text + "\n")
                output_file.write("-" * 40 + "\n\n")
                print(f"Extracted text from {f} saved to parsed.txt\n")
            else:
                print(f"Failed to parse {f}\n{'-'*40}\n")
    
    # --- Example with in-memory bytes (simulating API behavior) ---
    try:
        with open("dataset/d1.pdf", "rb") as f:
            pdf_bytes = f.read()
        print("Parsing in-memory PDF bytes...")
        in_memory_text = parse_document(pdf_bytes)
        if in_memory_text:
            print("In-memory parsing successful.\n" + "="*40)
        else:
            print("In-memory parsing failed.\n" + "="*40)
    except FileNotFoundError:
        print("Skipping in-memory test because dataset/d1.pdf was not found.")