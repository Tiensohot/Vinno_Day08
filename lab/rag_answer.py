"""
rag_answer.py — Sprint 2 + Sprint 3: Retrieval & Grounded Answer
================================================================
Sprint 2 (60 phút): Baseline RAG
  - Dense retrieval từ ChromaDB
  - Grounded answer function với prompt ép citation
  - Trả lời được ít nhất 3 câu hỏi mẫu, output có source

Sprint 3 (60 phút): Tuning tối thiểu
  - Thêm hybrid retrieval (dense + sparse/BM25) với Reciprocal Rank Fusion
  - Thêm rerank (cross-encoder)
  - Thêm query transformation (expansion, decomposition, HyDE)
  - Tạo bảng so sánh baseline vs variant

Definition of Done Sprint 2:
  ✓ rag_answer("SLA ticket P1?") trả về câu trả lời có citation
  ✓ rag_answer("Câu hỏi không có trong docs") trả về "Không đủ dữ liệu"

Definition of Done Sprint 3:
  ✓ Có ít nhất 1 variant (hybrid / rerank / query transform) chạy được
  ✓ Giải thích được tại sao chọn biến đó để tune
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CẤU HÌNH
# =============================================================================

TOP_K_SEARCH = 10    # Số chunk lấy từ vector store trước rerank (search rộng)
TOP_K_SELECT = 3     # Số chunk gửi vào prompt sau rerank/select (top-3 sweet spot)

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# Cache để tránh load lại nhiều lần
_bm25_index = None
_bm25_chunks = None
_rerank_model = None


# =============================================================================
# RETRIEVAL — DENSE (Vector Search)
# =============================================================================

def retrieve_dense(query: str, top_k: int = TOP_K_SEARCH) -> List[Dict[str, Any]]:
    """
    Dense retrieval: tìm kiếm theo embedding similarity trong ChromaDB.

    Args:
        query: Câu hỏi của người dùng
        top_k: Số chunk tối đa trả về

    Returns:
        List các dict, mỗi dict là một chunk với:
          - "text": nội dung chunk
          - "metadata": metadata (source, section, effective_date, ...)
          - "score": cosine similarity score (1 - distance)
    """
    import chromadb
    from index import get_embedding, CHROMA_DB_DIR

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = client.get_collection("rag_lab")

    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        # ChromaDB cosine distance: score = 1 - distance
        score = 1.0 - dist
        chunks.append({
            "text": doc,
            "metadata": meta,
            "score": score,
        })

    return chunks


# =============================================================================
# RETRIEVAL — SPARSE / BM25 (Keyword Search)
# =============================================================================

def _build_bm25_index() -> Tuple[Any, List[Dict[str, Any]]]:
    """
    Xây BM25 index từ toàn bộ chunks trong ChromaDB.
    Cache lại để không build lại mỗi lần query.
    """
    global _bm25_index, _bm25_chunks

    if _bm25_index is not None and _bm25_chunks is not None:
        return _bm25_index, _bm25_chunks

    from rank_bm25 import BM25Okapi
    import chromadb
    from index import CHROMA_DB_DIR

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = client.get_collection("rag_lab")

    # Lấy toàn bộ chunks
    results = collection.get(include=["documents", "metadatas"])
    all_docs = results["documents"]
    all_metas = results["metadatas"]

    _bm25_chunks = [
        {"text": doc, "metadata": meta}
        for doc, meta in zip(all_docs, all_metas)
    ]

    # Tokenize đơn giản: lowercase + split (hoạt động tốt với tiếng Việt không dấu
    # và tiếng Anh; đủ cho lab demo)
    tokenized_corpus = [chunk["text"].lower().split() for chunk in _bm25_chunks]
    _bm25_index = BM25Okapi(tokenized_corpus)

    return _bm25_index, _bm25_chunks


def retrieve_sparse(query: str, top_k: int = TOP_K_SEARCH) -> List[Dict[str, Any]]:
    """
    Sparse retrieval: tìm kiếm theo keyword (BM25).

    Mạnh ở: exact term, mã lỗi, tên riêng (ví dụ: "ERR-403", "P1", "refund")
    Hay hụt: câu hỏi paraphrase, đồng nghĩa

    Returns:
        List các dict với "text", "metadata", "score" (BM25 score, đã normalize 0-1)
    """
    bm25, all_chunks = _build_bm25_index()

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Lấy top_k theo score
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    max_score = scores[top_indices[0]] if top_indices else 1.0
    if max_score == 0:
        max_score = 1.0  # tránh chia 0

    chunks = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue
        chunks.append({
            "text": all_chunks[idx]["text"],
            "metadata": all_chunks[idx]["metadata"],
            "score": float(scores[idx]) / max_score,  # normalize 0-1
        })

    return chunks


# =============================================================================
# RETRIEVAL — HYBRID (Dense + Sparse với Reciprocal Rank Fusion)
# =============================================================================

def retrieve_hybrid(
    query: str,
    top_k: int = TOP_K_SEARCH,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval: kết hợp dense và sparse bằng Reciprocal Rank Fusion (RRF).

    RRF_score(doc) = dense_weight * (1 / (60 + dense_rank)) +
                     sparse_weight * (1 / (60 + sparse_rank))

    Mạnh ở: giữ được cả nghĩa (dense) lẫn keyword chính xác (sparse)
    Phù hợp khi: corpus lẫn lộn ngôn ngữ tự nhiên và tên riêng/mã lỗi/điều khoản
    """
    dense_results = retrieve_dense(query, top_k=top_k)
    sparse_results = retrieve_sparse(query, top_k=top_k)

    # Dùng text làm key để xác định unique chunks
    rrf_scores: Dict[str, float] = {}
    chunk_by_key: Dict[str, Dict[str, Any]] = {}

    k_rrf = 60  # hằng số RRF tiêu chuẩn

    for rank, chunk in enumerate(dense_results):
        key = chunk["text"][:200]  # dùng 200 ký tự đầu làm key
        rrf_scores[key] = rrf_scores.get(key, 0.0) + dense_weight * (1.0 / (k_rrf + rank))
        chunk_by_key[key] = chunk

    for rank, chunk in enumerate(sparse_results):
        key = chunk["text"][:200]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + sparse_weight * (1.0 / (k_rrf + rank))
        if key not in chunk_by_key:
            chunk_by_key[key] = chunk

    # Sort theo RRF score giảm dần
    sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)

    results = []
    for key in sorted_keys[:top_k]:
        chunk = dict(chunk_by_key[key])
        chunk["score"] = rrf_scores[key]  # gắn RRF score
        results.append(chunk)

    return results


# =============================================================================
# RERANK — Cross-Encoder
# =============================================================================

def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = TOP_K_SELECT,
) -> List[Dict[str, Any]]:
    """
    Rerank các candidate chunks bằng cross-encoder.

    Cross-encoder chấm lại "chunk nào thực sự trả lời câu hỏi này?"

    Funnel logic (từ slide):
      Search rộng (top-10) → Rerank (cross-encoder) → Select (top-3)

    Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (nhỏ gọn, chạy nhanh)
    """
    global _rerank_model

    if not candidates:
        return candidates

    try:
        from sentence_transformers import CrossEncoder

        if _rerank_model is None:
            print("[rerank] Loading cross-encoder model...")
            _rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        pairs = [[query, chunk["text"]] for chunk in candidates]
        scores = _rerank_model.predict(pairs)

        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

        results = []
        for chunk, score in ranked[:top_k]:
            chunk = dict(chunk)
            chunk["rerank_score"] = float(score)
            results.append(chunk)

        return results

    except ImportError:
        print("[rerank] sentence-transformers không có CrossEncoder, fallback về top_k đầu")
        return candidates[:top_k]
    except Exception as e:
        print(f"[rerank] Lỗi: {e} — fallback về top_k đầu")
        return candidates[:top_k]


# =============================================================================
# QUERY TRANSFORMATION
# =============================================================================

def transform_query(query: str, strategy: str = "expansion") -> List[str]:
    """
    Biến đổi query để tăng recall.

    Strategies:
      - "expansion": Thêm từ đồng nghĩa, alias, tên cũ → tăng recall
      - "decomposition": Tách query phức tạp thành 2-3 sub-queries
      - "hyde": Sinh câu trả lời giả (hypothetical document) để embed thay query

    Returns:
        List[str]: Danh sách queries (gốc + các biến thể)
    """
    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        if strategy == "expansion":
            prompt = (
                f"Given the search query: '{query}'\n"
                "Generate 2-3 alternative phrasings or related search terms "
                "in the SAME language as the query (Vietnamese or English).\n"
                "Include synonyms, abbreviations, and related concepts.\n"
                'Output ONLY a JSON array of strings, e.g.: ["alt1", "alt2"]'
            )
        elif strategy == "decomposition":
            prompt = (
                f"Break down this complex query into 2-3 simpler, more specific sub-queries: '{query}'\n"
                "Each sub-query should be self-contained and answerable independently.\n"
                'Output ONLY a JSON array of strings, e.g.: ["sub1", "sub2"]'
            )
        elif strategy == "hyde":
            prompt = (
                f"Write a concise, factual passage (2-3 sentences) that would be the ideal answer "
                f"to this question: '{query}'\n"
                "This is a hypothetical document for search purposes.\n"
                'Output ONLY a JSON array with one string: ["hypothetical answer text"]'
            )
        else:
            return [query]

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=256,
        )

        content = response.choices[0].message.content.strip()
        # Strip markdown code blocks nếu có
        if content.startswith("```"):
            content = "\n".join(content.split("\n")[1:])
        if content.endswith("```"):
            content = "\n".join(content.split("\n")[:-1])

        alternatives = json.loads(content)
        if isinstance(alternatives, list):
            # Luôn giữ query gốc ở đầu
            all_queries = [query] + [q for q in alternatives if q != query]
            return all_queries

    except Exception as e:
        print(f"[transform_query] Lỗi ({strategy}): {e} — dùng query gốc")

    return [query]


def retrieve_with_transform(
    query: str,
    strategy: str = "expansion",
    top_k: int = TOP_K_SEARCH,
    retrieval_fn=None,
) -> List[Dict[str, Any]]:
    """
    Retrieve với query transformation: chạy nhiều queries, merge và dedup kết quả.
    """
    if retrieval_fn is None:
        retrieval_fn = retrieve_dense

    queries = transform_query(query, strategy=strategy)
    print(f"[transform_query/{strategy}] Queries: {queries}")

    seen_texts: set = set()
    all_chunks: List[Dict[str, Any]] = []

    for q in queries:
        chunks = retrieval_fn(q, top_k=top_k)
        for chunk in chunks:
            key = chunk["text"][:200]
            if key not in seen_texts:
                seen_texts.add(key)
                all_chunks.append(chunk)

    # Sort theo score giảm dần (dùng score của query đầu tiên tìm được chunk đó)
    all_chunks.sort(key=lambda c: c.get("score", 0), reverse=True)
    return all_chunks[:top_k]


# =============================================================================
# GENERATION — GROUNDED ANSWER FUNCTION
# =============================================================================

def build_context_block(chunks: List[Dict[str, Any]]) -> str:
    """
    Đóng gói danh sách chunks thành context block để đưa vào prompt.

    Format: structured snippets với source, section, score.
    Mỗi chunk có số thứ tự [1], [2], ... để model dễ trích dẫn.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", "unknown")
        section = meta.get("section", "")
        effective_date = meta.get("effective_date", "")
        score = chunk.get("score", 0)
        text = chunk.get("text", "")

        header = f"[{i}] {source}"
        if section:
            header += f" | {section}"
        if effective_date and effective_date not in ("unknown", ""):
            header += f" | effective: {effective_date}"
        if score > 0:
            header += f" | score={score:.3f}"

        context_parts.append(f"{header}\n{text}")

    return "\n\n".join(context_parts)


def build_grounded_prompt(query: str, context_block: str) -> str:
    """
    Xây dựng grounded prompt theo 4 quy tắc:
    1. Evidence-only: Chỉ trả lời từ retrieved context
    2. Abstain: Thiếu context thì nói không đủ dữ liệu
    3. Citation: Gắn source/section khi có thể
    4. Short, clear, stable: Output ngắn, rõ, nhất quán
    """
    prompt = f"""Answer ONLY from the retrieved context below.
If the context does not contain enough information to answer the question,
say explicitly: "Không đủ dữ liệu trong tài liệu để trả lời câu hỏi này."
Do NOT make up information not present in the context.
Cite the source number (e.g. [1], [2]) when referencing specific content.
Keep your answer concise, clear, and factual.
Respond in the same language as the question.

Question: {query}

Context:
{context_block}

Answer:"""
    return prompt


def call_llm(prompt: str) -> str:
    """
    Gọi OpenAI ChatCompletion để sinh câu trả lời grounded.

    Dùng temperature=0 để output ổn định, dễ đánh giá.
    Fallback sang Gemini nếu OPENAI_API_KEY không có nhưng GOOGLE_API_KEY có.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if llm_provider == "gemini" or (not openai_key and google_key):
        # Fallback sang Gemini
        import google.generativeai as genai

        gemini_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        genai.configure(api_key=google_key)
        model = genai.GenerativeModel(gemini_model)
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0, "max_output_tokens": 512},
        )
        return response.text

    # Default: OpenAI
    from openai import OpenAI

    if not openai_key:
        raise ValueError(
            "Thiếu OPENAI_API_KEY trong .env. "
            "Set LLM_PROVIDER=gemini và GOOGLE_API_KEY để dùng Gemini thay thế."
        )

    client = OpenAI(api_key=openai_key)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=512,
    )
    return response.choices[0].message.content


def rag_answer(
    query: str,
    retrieval_mode: str = "dense",
    top_k_search: int = TOP_K_SEARCH,
    top_k_select: int = TOP_K_SELECT,
    use_rerank: bool = False,
    query_transform: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Pipeline RAG hoàn chỉnh: query → (transform) → retrieve → (rerank) → generate.

    Args:
        query: Câu hỏi
        retrieval_mode: "dense" | "sparse" | "hybrid"
        top_k_search: Số chunk lấy từ vector store (search rộng)
        top_k_select: Số chunk đưa vào prompt (sau rerank/select)
        use_rerank: Có dùng cross-encoder rerank không
        query_transform: None | "expansion" | "decomposition" | "hyde"
        verbose: In thêm thông tin debug

    Returns:
        Dict với:
          - "answer": câu trả lời grounded
          - "sources": list source names trích dẫn
          - "chunks_used": list chunks đã dùng
          - "query": query gốc
          - "config": cấu hình pipeline đã dùng
    """
    config = {
        "retrieval_mode": retrieval_mode,
        "top_k_search": top_k_search,
        "top_k_select": top_k_select,
        "use_rerank": use_rerank,
        "query_transform": query_transform,
    }

    # Chọn retrieval function
    retrieval_fn_map = {
        "dense": retrieve_dense,
        "sparse": retrieve_sparse,
        "hybrid": retrieve_hybrid,
    }
    if retrieval_mode not in retrieval_fn_map:
        raise ValueError(f"retrieval_mode không hợp lệ: {retrieval_mode}")

    retrieval_fn = retrieval_fn_map[retrieval_mode]

    # --- Bước 1: Retrieve (với hoặc không có query transformation) ---
    if query_transform:
        candidates = retrieve_with_transform(
            query,
            strategy=query_transform,
            top_k=top_k_search,
            retrieval_fn=retrieval_fn,
        )
    else:
        candidates = retrieval_fn(query, top_k=top_k_search)

    if verbose:
        print(f"\n[RAG] Query: {query}")
        print(f"[RAG] Retrieved {len(candidates)} candidates (mode={retrieval_mode}, transform={query_transform})")
        for i, c in enumerate(candidates[:5]):
            print(
                f"  [{i+1}] score={c.get('score', 0):.3f} | "
                f"{c['metadata'].get('source', '?')} | "
                f"{c['metadata'].get('section', '')[:40]}"
            )

    # --- Bước 2: Rerank hoặc truncate ---
    if use_rerank:
        candidates = rerank(query, candidates, top_k=top_k_select)
        if verbose:
            print(f"[RAG] After rerank: {len(candidates)} chunks")
    else:
        candidates = candidates[:top_k_select]
        if verbose:
            print(f"[RAG] After select (no rerank): {len(candidates)} chunks")

    # --- Bước 3: Build context và prompt ---
    context_block = build_context_block(candidates)
    prompt = build_grounded_prompt(query, context_block)

    if verbose:
        print(f"\n[RAG] Context block:\n{context_block[:600]}...")
        print(f"\n[RAG] Calling LLM ({LLM_MODEL})...")

    # --- Bước 4: Generate ---
    answer = call_llm(prompt)

    # --- Bước 5: Extract sources ---
    sources = list({
        c["metadata"].get("source", "unknown")
        for c in candidates
    })

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "chunks_used": candidates,
        "config": config,
    }


# =============================================================================
# SPRINT 3: SO SÁNH BASELINE VS VARIANT
# =============================================================================

def compare_retrieval_strategies(query: str, verbose: bool = False) -> None:
    """
    So sánh các retrieval strategies với cùng một query.
    A/B Rule: Chỉ đổi MỘT biến mỗi lần.
    """
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print("=" * 60)

    experiments = [
        {"label": "Baseline (Dense, no rerank)",    "mode": "dense",  "rerank": False, "transform": None},
        {"label": "Variant A: Hybrid RRF",           "mode": "hybrid", "rerank": False, "transform": None},
        {"label": "Variant B: Dense + Rerank",       "mode": "dense",  "rerank": True,  "transform": None},
        {"label": "Variant C: Dense + Query Expand", "mode": "dense",  "rerank": False, "transform": "expansion"},
    ]

    for exp in experiments:
        print(f"\n--- {exp['label']} ---")
        try:
            result = rag_answer(
                query,
                retrieval_mode=exp["mode"],
                use_rerank=exp["rerank"],
                query_transform=exp["transform"],
                verbose=verbose,
            )
            print(f"Answer  : {result['answer']}")
            print(f"Sources : {result['sources']}")
        except NotImplementedError as e:
            print(f"Chưa implement: {e}")
        except Exception as e:
            print(f"Lỗi: {type(e).__name__}: {e}")


# =============================================================================
# MAIN — Demo và Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sprint 2 + 3: RAG Answer Pipeline")
    print("=" * 60)

    # Test queries từ data/test_questions.json
    test_queries = [
        "SLA xử lý ticket P1 là bao lâu?",
        "Khách hàng có thể yêu cầu hoàn tiền trong bao nhiêu ngày?",
        "Ai phải phê duyệt để cấp quyền Level 3?",
        "ERR-403-AUTH là lỗi gì?",         # → kiểm tra abstain
        "Nhân viên được làm remote mấy ngày?",
    ]

    print("\n--- Sprint 2: Test Baseline (Dense) ---")
    for query in test_queries:
        print(f"\n{'-'*50}")
        print(f"Query: {query}")
        try:
            result = rag_answer(query, retrieval_mode="dense", verbose=True)
            print(f"\nAnswer  : {result['answer']}")
            print(f"Sources : {result['sources']}")
        except Exception as e:
            print(f"Lỗi: {type(e).__name__}: {e}")

    # Sprint 3: So sánh strategies
    print("\n\n--- Sprint 3: So sánh strategies ---")
    compare_retrieval_strategies("Approval Matrix để cấp quyền hệ thống là tài liệu nào?")
    compare_retrieval_strategies("ERR-403-AUTH")
