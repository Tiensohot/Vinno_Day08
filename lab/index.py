"""
index.py — Sprint 1: Build RAG Index
====================================

Mục tiêu Sprint 1 (60 phút):
- Đọc và preprocess tài liệu từ data/docs/
- Chunk tài liệu theo cấu trúc tự nhiên (heading/section)
- Gắn metadata: source, section, department, effective_date, access
- Embed và lưu vào vector store (ChromaDB)

Definition of Done Sprint 1:
✓ Script chạy được và index đủ docs
✓ Có ít nhất 3 metadata fields hữu ích cho retrieval
✓ Có thể kiểm tra chunk bằng list_chunks()
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CẤU HÌNH
# =============================================================================

DOCS_DIR = Path(__file__).parent / "data" / "docs"
CHROMA_DB_DIR = Path(__file__).parent / "chroma_db"

# Gợi ý từ slide: chunk 300-500 tokens, overlap 50-80 tokens
# 1 token ~ 4 ký tự (ước lượng thô)
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

LOCAL_EMBED_MODEL = os.getenv(
    "LOCAL_EMBED_MODEL",
    "paraphrase-multilingual-MiniLM-L12-v2",
)
OPENAI_EMBED_MODEL = os.getenv(
    "OPENAI_EMBED_MODEL",
    "text-embedding-3-small",
)
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").lower()

# Cache model/client để không load lại nhiều lần
_embedding_model = None
_openai_client = None


# =============================================================================
# HELPERS
# =============================================================================

def _normalize_whitespace(text: str) -> str:
    """Chuẩn hóa khoảng trắng nhưng vẫn giữ cấu trúc đoạn văn."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\ufeff", "")
    text = "\n".join(re.sub(r"[ \t]+", " ", line).rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunk_len(units: List[str]) -> int:
    """Độ dài khi ghép units bằng \\n\\n."""
    if not units:
        return 0
    return sum(len(u) for u in units) + (len(units) - 1) * 2


def _split_long_unit(unit: str, max_chars: int) -> List[str]:
    """
    Cắt một đoạn quá dài theo sentence trước, nếu vẫn quá dài thì hard-split.
    """
    unit = unit.strip()
    if not unit:
        return []

    if len(unit) <= max_chars:
        return [unit]

    # Thử tách theo câu / mốc tự nhiên
    sentences = re.split(r"(?<=[\.\!\?\:\;])\s+", unit)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= 1:
        # Hard split nếu không tách câu được
        pieces = []
        start = 0
        while start < len(unit):
            end = min(start + max_chars, len(unit))
            piece = unit[start:end]

            if end < len(unit):
                natural_breaks = [piece.rfind("\n"), piece.rfind(". "), piece.rfind("; "), piece.rfind(", ")]
                best = max(natural_breaks)
                if best > int(max_chars * 0.6):
                    end = start + best + 1
                    piece = unit[start:end]

            pieces.append(piece.strip())
            if end <= start:
                break
            start = end

        return [p for p in pieces if p]

    chunks: List[str] = []
    current = ""

    for sent in sentences:
        if len(sent) > max_chars:
            if current.strip():
                chunks.append(current.strip())
                current = ""
            chunks.extend(_split_long_unit(sent, max_chars))
            continue

        candidate = sent if not current else f"{current} {sent}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current.strip())
            current = sent

    if current.strip():
        chunks.append(current.strip())

    return chunks


def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chroma chỉ chấp nhận scalar types trong metadata.
    """
    safe: Dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            safe[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            safe[key] = value
        else:
            safe[key] = str(value)
    return safe


# =============================================================================
# STEP 1: PREPROCESS
# Làm sạch text trước khi chunk và embed
# =============================================================================

def preprocess_document(raw_text: str, filepath: str) -> Dict[str, Any]:
    """
    Preprocess một tài liệu: extract metadata từ header và làm sạch nội dung.

    Args:
        raw_text: Toàn bộ nội dung file text
        filepath: Đường dẫn file để làm source mặc định

    Returns:
        Dict chứa:
        - "text": nội dung đã clean
        - "metadata": dict với source, department, effective_date, access
    """
    raw_text = _normalize_whitespace(raw_text)
    lines = raw_text.split("\n")

    metadata = {
        "source": Path(filepath).name,
        "section": "",
        "department": "unknown",
        "effective_date": "unknown",
        "access": "internal",
    }

    patterns = {
        "source": re.compile(r"^\s*Source:\s*(.+?)\s*$", re.IGNORECASE),
        "department": re.compile(r"^\s*Department:\s*(.+?)\s*$", re.IGNORECASE),
        "effective_date": re.compile(r"^\s*Effective Date:\s*(.+?)\s*$", re.IGNORECASE),
        "access": re.compile(r"^\s*Access:\s*(.+?)\s*$", re.IGNORECASE),
    }

    content_lines: List[str] = []
    header_done = False

    for line in lines:
        stripped = line.strip()

        if not header_done:
            matched_meta = False
            for key, pattern in patterns.items():
                m = pattern.match(stripped)
                if m:
                    metadata[key] = m.group(1).strip()
                    matched_meta = True
                    break

            if matched_meta:
                continue

            if not stripped:
                continue

            # Bỏ dòng title ở đầu file (thường viết hoa toàn bộ)
            is_probable_title = (
                ":" not in stripped
                and not stripped.startswith("===")
                and re.fullmatch(r"[A-ZÀ-Ỹ0-9\s\-\(\)\/\.]+", stripped) is not None
                and len(stripped) > 8
            )
            if is_probable_title:
                continue

            # Từ đây trở đi coi như nội dung thật
            header_done = True

        content_lines.append(stripped)

    cleaned_text = "\n".join(content_lines)
    cleaned_text = _normalize_whitespace(cleaned_text)

    return {
        "text": cleaned_text,
        "metadata": metadata,
    }


# =============================================================================
# STEP 2: CHUNK
# Chia tài liệu thành các đoạn nhỏ theo cấu trúc tự nhiên
# =============================================================================

def chunk_document(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Chunk một tài liệu đã preprocess thành danh sách các chunk nhỏ.

    Args:
        doc: Dict với "text" và "metadata" (output của preprocess_document)

    Returns:
        List các Dict, mỗi dict là một chunk với:
        - "text": nội dung chunk
        - "metadata": metadata gốc + "section" của chunk đó
    """
    text = doc["text"]
    base_metadata = doc["metadata"].copy()
    chunks: List[Dict[str, Any]] = []

    heading_pattern = re.compile(r"(?m)^===\s*.+?\s*===\s*$")
    matches = list(heading_pattern.finditer(text))

    if not matches:
        return _split_by_size(
            text=text,
            base_metadata=base_metadata,
            section="General",
        )

    # Nội dung trước heading đầu tiên, ví dụ "Ghi chú: ..."
    preface = text[: matches[0].start()].strip()
    if preface:
        chunks.extend(
            _split_by_size(
                text=preface,
                base_metadata=base_metadata,
                section="General",
            )
        )

    for i, match in enumerate(matches):
        section_title = match.group(0).strip("= ").strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()

        if not section_text:
            continue

        chunks.extend(
            _split_by_size(
                text=section_text,
                base_metadata=base_metadata,
                section=section_title,
            )
        )

    return chunks


def _split_by_size(
    text: str,
    base_metadata: Dict[str, Any],
    section: str,
    chunk_chars: int = CHUNK_SIZE * 4,
    overlap_chars: int = CHUNK_OVERLAP * 4,
) -> List[Dict[str, Any]]:
    """
    Split text dài thành chunks theo paragraph trước, sau đó theo sentence,
    có overlap tự nhiên ở cuối chunk trước.
    """
    text = _normalize_whitespace(text)
    if not text:
        return []

    if len(text) <= chunk_chars:
        return [
            {
                "text": text,
                "metadata": {
                    **base_metadata,
                    "section": section,
                },
            }
        ]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    units: List[str] = []
    for para in paragraphs:
        units.extend(_split_long_unit(para, chunk_chars))

    chunk_dicts: List[Dict[str, Any]] = []
    current_units: List[str] = []

    for unit in units:
        if not current_units:
            current_units = [unit]
            continue

        candidate_units = current_units + [unit]
        if _chunk_len(candidate_units) <= chunk_chars:
            current_units = candidate_units
            continue

        # Flush chunk hiện tại
        chunk_text = "\n\n".join(current_units).strip()
        chunk_dicts.append(
            {
                "text": chunk_text,
                "metadata": {
                    **base_metadata,
                    "section": section,
                },
            }
        )

        # Tạo overlap từ cuối chunk trước
        overlap_units: List[str] = []
        overlap_len = 0
        for prev in reversed(current_units):
            prev_len = len(prev) + (2 if overlap_units else 0)
            if overlap_units and overlap_len + prev_len > overlap_chars:
                break
            overlap_units.insert(0, prev)
            overlap_len += prev_len
            if overlap_len >= overlap_chars:
                break

        current_units = overlap_units + [unit]

        # Trường hợp hiếm: overlap + unit vẫn vượt max quá nhiều
        while _chunk_len(current_units) > chunk_chars and len(current_units) > 1:
            current_units.pop(0)

    if current_units:
        chunk_text = "\n\n".join(current_units).strip()
        chunk_dicts.append(
            {
                "text": chunk_text,
                "metadata": {
                    **base_metadata,
                    "section": section,
                },
            }
        )

    # Loại duplicate liên tiếp nếu overlap làm trùng nguyên chunk
    deduped: List[Dict[str, Any]] = []
    prev_text = None
    for chunk in chunk_dicts:
        if chunk["text"] != prev_text:
            deduped.append(chunk)
        prev_text = chunk["text"]

    return deduped


# =============================================================================
# STEP 3: EMBED + STORE
# Embed các chunk và lưu vào ChromaDB
# =============================================================================

def get_embedding(text: str) -> List[float]:
    """
    Tạo embedding vector cho một đoạn text.

    Mặc định dùng Sentence Transformers để chạy local, không cần API key.
    Có thể chuyển sang OpenAI bằng cách set:
        EMBEDDING_PROVIDER=openai
        OPENAI_API_KEY=...
    """
    global _embedding_model, _openai_client

    text = text.strip()
    if not text:
        raise ValueError("Text rỗng, không thể tạo embedding.")

    provider = EMBEDDING_PROVIDER

    if provider == "openai":
        from openai import OpenAI

        if _openai_client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "EMBEDDING_PROVIDER=openai nhưng thiếu OPENAI_API_KEY."
                )
            _openai_client = OpenAI(api_key=api_key)

        response = _openai_client.embeddings.create(
            input=text,
            model=OPENAI_EMBED_MODEL,
        )
        return response.data[0].embedding

    # Local by default
    from sentence_transformers import SentenceTransformer

    if _embedding_model is None:
        _embedding_model = SentenceTransformer(LOCAL_EMBED_MODEL)

    vector = _embedding_model.encode(
        text,
        normalize_embeddings=True,
    )
    return vector.tolist()


def build_index(docs_dir: Path = DOCS_DIR, db_dir: Path = CHROMA_DB_DIR) -> None:
    """
    Pipeline hoàn chỉnh: đọc docs → preprocess → chunk → embed → store.
    """
    import chromadb

    print(f"Đang build index từ: {docs_dir}")
    db_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(db_dir))

    collection_name = "rag_lab"

    # Reset collection cũ để tránh duplicate khi chạy nhiều lần
    try:
        client.delete_collection(collection_name)
        print(f"Đã xóa collection cũ: {collection_name}")
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    doc_files = sorted(docs_dir.glob("*.txt"))
    if not doc_files:
        print(f"Không tìm thấy file .txt trong {docs_dir}")
        return

    total_chunks = 0

    for filepath in doc_files:
        print(f"\nProcessing: {filepath.name}")
        raw_text = filepath.read_text(encoding="utf-8")

        doc = preprocess_document(raw_text, str(filepath))
        chunks = chunk_document(doc)

        if not chunks:
            print(" -> Bỏ qua: không tạo được chunk nào")
            continue

        ids: List[str] = []
        embeddings: List[List[float]] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{filepath.stem}_{i:03d}"
            chunk_text = chunk["text"]
            chunk_meta = {
                **chunk["metadata"],
                "chunk_id": chunk_id,
                "chunk_index": i,
                "char_len": len(chunk_text),
            }

            embedding = get_embedding(chunk_text)

            ids.append(chunk_id)
            embeddings.append(embedding)
            documents.append(chunk_text)
            metadatas.append(_sanitize_metadata(chunk_meta))

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        print(f" -> {len(chunks)} chunks indexed")
        total_chunks += len(chunks)

    print(f"\nHoàn thành! Tổng số chunks đã index: {total_chunks}")
    print(f"Collection: {collection_name}")
    print(f"DB path: {db_dir}")


# =============================================================================
# STEP 4: INSPECT / KIỂM TRA
# Dùng để debug và kiểm tra chất lượng index
# =============================================================================

def list_chunks(db_dir: Path = CHROMA_DB_DIR, n: int = 5) -> None:
    """
    In ra n chunk đầu tiên trong ChromaDB để kiểm tra chất lượng index.
    """
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(db_dir))
        collection = client.get_collection("rag_lab")
        results = collection.get(limit=n, include=["documents", "metadatas"])

        print(f"\n=== Top {n} chunks trong index ===\n")
        for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
            print(f"[Chunk {i + 1}]")
            print(f" Source: {meta.get('source', 'N/A')}")
            print(f" Section: {meta.get('section', 'N/A')}")
            print(f" Effective Date: {meta.get('effective_date', 'N/A')}")
            print(f" Department: {meta.get('department', 'N/A')}")
            print(f" Text preview: {doc[:180]}...")
            print()
    except Exception as e:
        print(f"Lỗi khi đọc index: {e}")
        print("Hãy chạy build_index() trước.")


def inspect_metadata_coverage(db_dir: Path = CHROMA_DB_DIR) -> None:
    """
    Kiểm tra phân phối metadata trong toàn bộ index.
    """
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(db_dir))
        collection = client.get_collection("rag_lab")
        results = collection.get(include=["metadatas"])

        metadatas = results["metadatas"]
        print(f"\nTổng chunks: {len(metadatas)}")

        departments: Dict[str, int] = {}
        sources: Dict[str, int] = {}
        missing_date = 0
        missing_section = 0

        for meta in metadatas:
            dept = meta.get("department", "unknown")
            src = meta.get("source", "unknown")
            section = meta.get("section", "")

            departments[dept] = departments.get(dept, 0) + 1
            sources[src] = sources.get(src, 0) + 1

            if meta.get("effective_date") in ("unknown", "", None):
                missing_date += 1
            if section in ("", None):
                missing_section += 1

        print("\nPhân bố theo department:")
        for dept, count in sorted(departments.items(), key=lambda x: x[0]):
            print(f" - {dept}: {count} chunks")

        print("\nPhân bố theo source:")
        for src, count in sorted(sources.items(), key=lambda x: x[0]):
            print(f" - {src}: {count} chunks")

        print(f"\nChunks thiếu effective_date: {missing_date}")
        print(f"Chunks thiếu section: {missing_section}")

    except Exception as e:
        print(f"Lỗi: {e}")
        print("Hãy chạy build_index() trước.")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sprint 1: Build RAG Index")
    print("=" * 60)

    doc_files = sorted(DOCS_DIR.glob("*.txt"))
    print(f"\nTìm thấy {len(doc_files)} tài liệu:")
    for f in doc_files:
        print(f" - {f.name}")

    # Bước 1: Test preprocess + chunking trước
    print("\n--- Test preprocess + chunking ---")
    if doc_files:
        filepath = doc_files[0]
        raw = filepath.read_text(encoding="utf-8")
        doc = preprocess_document(raw, str(filepath))
        chunks = chunk_document(doc)

        print(f"\nFile: {filepath.name}")
        print(f"Metadata: {doc['metadata']}")
        print(f"Số chunks: {len(chunks)}")

        for i, chunk in enumerate(chunks[:3]):
            print(f"\n[Chunk {i + 1}] Section: {chunk['metadata']['section']}")
            print(f"Text: {chunk['text'][:180]}...")

    # Bước 2: Build full index
    print("\n--- Build Full Index ---")
    print(
        "Ghi chú: lần chạy đầu với sentence-transformers có thể tải model về máy."
    )
    build_index()

    # Bước 3: Kiểm tra index
    print("\n--- Inspect Index ---")
    list_chunks()
    inspect_metadata_coverage()

    print("\nSprint 1 hoàn thành!")