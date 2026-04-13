# -*- coding: utf-8 -*-
"""
test_rag.py — Kiểm tra toàn bộ rag_answer.py (Sprint 2 + Sprint 3)
Chạy: python test_rag.py
"""
import sys
import json
from pathlib import Path

# Đảm bảo stdout nhận UTF-8 trên Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from rag_answer import (
    retrieve_dense,
    retrieve_sparse,
    retrieve_hybrid,
    rerank,
    transform_query,
    rag_answer,
    compare_retrieval_strategies,
)

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"
SEP  = "=" * 60


def check(label: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    print(f"  {status} {label}" + (f" — {detail}" if detail else ""))
    return condition


# =============================================================================
# SPRINT 2 — BASELINE DENSE
# =============================================================================

def test_sprint2():
    print(f"\n{SEP}")
    print("SPRINT 2: Baseline Dense Retrieval + Grounded Answer")
    print(SEP)

    results = {}

    # --- Test 1: Dense retrieval trả về chunks ---
    print("\n[1] retrieve_dense()")
    try:
        chunks = retrieve_dense("SLA ticket P1", top_k=5)
        check("Trả về list", isinstance(chunks, list))
        check("Có ít nhất 1 chunk", len(chunks) >= 1)
        check("Chunk có key 'text'", "text" in chunks[0])
        check("Chunk có key 'metadata'", "metadata" in chunks[0])
        check("Chunk có key 'score'", "score" in chunks[0])
        check("Score trong [0,1]", 0 <= chunks[0]["score"] <= 1,
              f"score={chunks[0]['score']:.4f}")
        results["dense_ok"] = True
    except Exception as e:
        print(f"  {FAIL} Exception: {e}")
        results["dense_ok"] = False

    # --- Test 2: rag_answer trả lời đúng với citation ---
    print("\n[2] rag_answer() — câu hỏi có trong docs")
    qa_cases = [
        {
            "query": "SLA xử lý ticket P1 là bao lâu?",
            "expected_keyword": "4",       # 4 giờ
            "expected_source_hint": "sla",
        },
        {
            "query": "Khách hàng có thể yêu cầu hoàn tiền trong bao nhiêu ngày?",
            "expected_keyword": "7",       # 7 ngày
            "expected_source_hint": "refund",
        },
        {
            "query": "Ai phải phê duyệt để cấp quyền Level 3?",
            "expected_keyword": "IT Security",
            "expected_source_hint": "access",
        },
        {
            "query": "Nhân viên được làm remote mấy ngày mỗi tuần?",
            "expected_keyword": "2",
            "expected_source_hint": "leave",
        },
    ]

    for case in qa_cases:
        print(f"\n  Query: {case['query']}")
        try:
            result = rag_answer(case["query"], retrieval_mode="dense", verbose=False)
            answer  = result["answer"]
            sources = result["sources"]

            check("Có answer", bool(answer))
            check("Có sources", len(sources) > 0)
            check(
                "Answer chứa keyword kỳ vọng",
                case["expected_keyword"].lower() in answer.lower(),
                f"keyword='{case['expected_keyword']}' | answer='{answer[:80]}'"
            )
            check(
                "Source đúng tài liệu",
                any(case["expected_source_hint"].lower() in s.lower() for s in sources),
                f"sources={sources}"
            )
            # Citation check: có [1] hay [2] trong answer
            has_citation = any(f"[{n}]" in answer for n in range(1, 6))
            check("Answer có citation [n]", has_citation, f"answer='{answer[:80]}'")
        except Exception as e:
            print(f"  {FAIL} Exception: {e}")

    # --- Test 3: Abstain khi không có thông tin ---
    print("\n[3] rag_answer() — câu hỏi KHÔNG có trong docs (phải abstain)")
    abstain_queries = [
        "ERR-403-AUTH là lỗi gì?",
        "Tên CEO của công ty là gì?",
    ]
    for q in abstain_queries:
        print(f"\n  Query: {q}")
        try:
            result = rag_answer(q, retrieval_mode="dense", verbose=False)
            answer = result["answer"]
            print(f"  Answer: {answer}")
            # Model nên nói "Không đủ dữ liệu" hoặc "I don't know" dạng gì đó
            abstain_signals = [
                "không đủ dữ liệu", "không có thông tin",
                "not enough", "i don't know", "do not know",
                "không tìm thấy", "không có trong tài liệu"
            ]
            abstained = any(sig in answer.lower() for sig in abstain_signals)
            check("Model abstain (không bịa)", abstained,
                  "Nếu FAIL: model đang hallucinate — review prompt")
        except Exception as e:
            print(f"  {FAIL} Exception: {e}")

    print(f"\n{SEP}")
    print("Sprint 2 tests DONE")


# =============================================================================
# SPRINT 3 — VARIANTS
# =============================================================================

def test_sprint3():
    print(f"\n{SEP}")
    print("SPRINT 3: Variant Retrieval Strategies")
    print(SEP)

    test_query = "Approval Matrix để cấp quyền hệ thống là tài liệu nào?"

    # --- Variant A: BM25 Sparse ---
    print("\n[Variant A] BM25 Sparse retrieval")
    try:
        chunks = retrieve_sparse(test_query, top_k=5)
        check("BM25 trả về list", isinstance(chunks, list))
        check("Có ít nhất 1 chunk", len(chunks) >= 1, f"got {len(chunks)} chunks")
        if chunks:
            check("Score normalize <= 1", chunks[0]["score"] <= 1.0,
                  f"score={chunks[0]['score']:.4f}")
            print(f"  Top-3 sparse chunks:")
            for i, c in enumerate(chunks[:3], 1):
                print(f"    [{i}] score={c['score']:.3f} | {c['metadata'].get('source')} | {c['metadata'].get('section','')[:40]}")
    except Exception as e:
        print(f"  {FAIL} Exception: {e}")

    # --- Variant A: Hybrid RRF ---
    print("\n[Variant A] Hybrid RRF retrieval")
    try:
        chunks = retrieve_hybrid(test_query, top_k=5)
        check("Hybrid trả về list", isinstance(chunks, list))
        check("Có ít nhất 1 chunk", len(chunks) >= 1)
        if chunks:
            print(f"  Top-3 hybrid (RRF) chunks:")
            for i, c in enumerate(chunks[:3], 1):
                print(f"    [{i}] rrf_score={c['score']:.4f} | {c['metadata'].get('source')} | {c['metadata'].get('section','')[:40]}")
    except Exception as e:
        print(f"  {FAIL} Exception: {e}")

    # rag_answer với hybrid
    print("\n[Variant A] rag_answer(hybrid)")
    try:
        result = rag_answer(test_query, retrieval_mode="hybrid", verbose=False)
        answer  = result["answer"]
        sources = result["sources"]
        print(f"  Answer  : {answer}")
        print(f"  Sources : {sources}")
        check("Có answer", bool(answer))
        check("Nhắc đến 'access-control' hoặc 'Approval'",
              any(kw.lower() in answer.lower() for kw in ["approval", "access", "sop", "matrix"]),
              f"answer='{answer[:80]}'")
    except Exception as e:
        print(f"  {FAIL} Exception: {e}")

    # --- Variant B: Dense + Rerank ---
    print("\n[Variant B] Dense + Cross-Encoder Rerank")
    try:
        candidates = retrieve_dense(test_query, top_k=10)
        reranked   = rerank(test_query, candidates, top_k=3)
        check("Rerank trả về list", isinstance(reranked, list))
        check("Top-k <= 3", len(reranked) <= 3)
        if reranked:
            check("Có 'rerank_score' trong chunk", "rerank_score" in reranked[0])
            print(f"  Top-3 reranked chunks:")
            for i, c in enumerate(reranked, 1):
                print(f"    [{i}] rerank_score={c.get('rerank_score', 0):.3f} | {c['metadata'].get('source')} | {c['metadata'].get('section','')[:40]}")
    except Exception as e:
        print(f"  {FAIL} Exception: {e}")

    # rag_answer với rerank
    print("\n[Variant B] rag_answer(dense + rerank)")
    try:
        result = rag_answer(test_query, retrieval_mode="dense", use_rerank=True, verbose=False)
        print(f"  Answer  : {result['answer']}")
        print(f"  Sources : {result['sources']}")
        check("Có answer", bool(result["answer"]))
    except Exception as e:
        print(f"  {FAIL} Exception: {e}")

    # --- Variant C: Query Transformation (Expansion) ---
    print("\n[Variant C] Query Expansion")
    try:
        expanded = transform_query(test_query, strategy="expansion")
        check("Trả về list", isinstance(expanded, list))
        check("Có ít nhất 2 queries (gốc + expansion)", len(expanded) >= 2,
              f"got {len(expanded)} queries")
        print(f"  Expanded queries:")
        for i, q in enumerate(expanded, 1):
            print(f"    [{i}] {q}")
    except Exception as e:
        print(f"  {FAIL} Exception: {e}")

    # rag_answer với query expansion
    print("\n[Variant C] rag_answer(dense + query_transform=expansion)")
    try:
        result = rag_answer(
            test_query,
            retrieval_mode="dense",
            query_transform="expansion",
            verbose=False,
        )
        print(f"  Answer  : {result['answer']}")
        print(f"  Sources : {result['sources']}")
        check("Có answer", bool(result["answer"]))
    except Exception as e:
        print(f"  {FAIL} Exception: {e}")

    # --- Full compare ---
    print(f"\n{SEP}")
    print("FULL COMPARISON — compare_retrieval_strategies()")
    print(SEP)
    compare_retrieval_strategies("Approval Matrix để cấp quyền hệ thống là tài liệu nào?")
    compare_retrieval_strategies("ERR-403-AUTH là lỗi gì và cách xử lý?")

    print(f"\n{SEP}")
    print("Sprint 3 tests DONE")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(SEP)
    print("RAG ANSWER TEST SUITE — Sprint 2 + Sprint 3")
    print(SEP)

    test_sprint2()
    test_sprint3()

    print(f"\n{SEP}")
    print("ALL TESTS DONE")
    print(SEP)
