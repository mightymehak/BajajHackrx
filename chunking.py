import re
from typing import List, Dict, Optional
import spacy
import json

# Load SpaCy English model once
nlp = spacy.load("en_core_web_sm")


def clean_text(text: str) -> str:
    text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\f', '', text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    text = text.replace('\r', '')
    return text.strip()


def detect_sections(text: str) -> List[Dict]:
    heading_regex = re.compile(
        r'((Section|Clause|Article|Chapter|Part) *\d+(\.\d+)*[:.]?|^\d+(\.\d+)+[\.\)]?)\s*[^\n]+\n',
        re.IGNORECASE | re.MULTILINE)
    sections = []
    indices = [m.start() for m in heading_regex.finditer(text)]
    if not indices:
        return [{'section_title': 'Document Start', 'section_text': text.strip()}]
    for i, start_idx in enumerate(indices):
        end_idx = indices[i + 1] if i + 1 < len(indices) else len(text)
        heading_line_match = heading_regex.match(text, pos=start_idx)
        section_title = heading_line_match.group().strip() if heading_line_match else f"Section {i+1}"
        section_text = text[start_idx:end_idx].strip()
        sections.append({'section_title': section_title, 'section_text': section_text})
    return sections


def split_into_sentences_paragraphs(text: str) -> List[str]:
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
    chunks = []
    buffer = []
    for sent in sentences:
        if re.match(r'^([-*•\d]+[\).\s]|[a-z]\))', sent):
            if buffer:
                chunks.append(' '.join(buffer))
                buffer = []
            chunks.append(sent)
        else:
            buffer.append(sent)
    if buffer:
        chunks.append(' '.join(buffer))
    return chunks


def filter_irrelevant(chunk_text: str) -> bool:
    chunk_text_lower = chunk_text.lower()
    boilerplate_phrases = [
        "this page intentionally left blank",
        "page number", "copyright", "confidential", "disclaimer"
    ]
    if any(phrase in chunk_text_lower for phrase in boilerplate_phrases):
        return False
    if len(chunk_text.strip()) < 30:
        return False
    return True


def tag_domains(chunk_text: str) -> Optional[str]:
    text = chunk_text.lower()
    if 'exclusion' in text or 'exclude' in text or 'not covered' in text:
        return 'exclusion'
    if 'coverage' in text or 'included' in text or 'benefit' in text:
        return 'coverage'
    if 'claim process' in text or 'claim' in text or 'procedure' in text:
        return 'claims'
    if 'eligibility' in text or 'eligib' in text:
        return 'eligibility'
    return None


def split_long_chunk(chunk_text: str, max_words: int = 300) -> List[str]:
    """
    Split large text chunk into smaller chunks based on sentence boundaries,
    ensuring each chunk is under max_words length.
    """
    doc = nlp(chunk_text)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 0]

    smaller_chunks = []
    current_chunk = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent.split())
        if current_len + sent_len > max_words and current_chunk:
            smaller_chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_len = 0
        current_chunk.append(sent)
        current_len += sent_len

    if current_chunk:
        smaller_chunks.append(' '.join(current_chunk))

    return smaller_chunks


def chunk_document(text: str, doc_name: str, overlap: int = 1,
                   max_chunk_words: int = 300) -> List[Dict]:
    cleaned = clean_text(text)
    sections = detect_sections(cleaned)
    chunks = []
    chunk_id = 1

    for section in sections:
        section_title = section['section_title']
        section_body = section['section_text'].replace(section_title, '', 1).strip()
        sentences_paragraphs = split_into_sentences_paragraphs(section_body)

        for i in range(len(sentences_paragraphs)):
            start_idx = max(0, i - overlap)
            chunk_text = ' '.join(sentences_paragraphs[start_idx:i + 1]).strip()

            if not filter_irrelevant(chunk_text):
                continue

            # Split chunk if too large
            if len(chunk_text.split()) > max_chunk_words:
                smaller_chunks = split_long_chunk(chunk_text, max_words=max_chunk_words)
            else:
                smaller_chunks = [chunk_text]

            for small_chunk_text in smaller_chunks:
                domain_tag = tag_domains(small_chunk_text)
                confidence = 'high' if len(small_chunk_text.split()) > 40 else 'low'
                chunks.append({
                    'doc_name': doc_name,
                    'chunk_id': chunk_id,
                    'section_title': section_title,
                    'chunk_text': small_chunk_text,
                    'chunk_length': len(small_chunk_text.split()),
                    'domain_tag': domain_tag,
                    'confidence': confidence
                })
                chunk_id += 1

    return chunks


if __name__ == "__main__":
    with open("parsed.txt", "r", encoding="utf-8") as file:
        document_text = file.read()

    document_name = "dataset/d1.pdf"
    chunked = chunk_document(document_text, document_name, overlap=1, max_chunk_words=300)

    with open("chunked_output.json", "w", encoding="utf-8") as f:
        json.dump(chunked, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(chunked)} chunks. Saved to chunked_output.json\n")

    for c in chunked[:5]:
        print(f"Chunk ID: {c['chunk_id']} | Section: {c['section_title']} | Domain: {c['domain_tag']} | Confidence: {c['confidence']}")
        print(f"Text (first 150 chars): {c['chunk_text'][:150]}...\n{'='*60}\n")
