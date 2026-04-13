# Tuning Log — RAG Pipeline (Day 08 Lab)

> Template: Ghi lại mỗi thay đổi và kết quả quan sát được.
> A/B Rule: Chỉ đổi MỘT biến mỗi lần.

---

## Baseline (Sprint 2)

**Ngày:** 2026-04-13  
**Config:**
```
retrieval_mode  = "dense"
chunk_size      = 400 ký tự (~100 tokens)
overlap         = 80 ký tự
top_k_search    = 10
top_k_select    = 3
use_rerank      = False
embedding_model = text-embedding-3-small (OpenAI)
llm_model       = gpt-4o-mini
```

**Scorecard Baseline:**
| Metric | Average Score |
|--------|--------------|
| Faithfulness | 4.60 /5 |
| Answer Relevance | 4.20 /5 |
| Context Recall | 4.44 /5 |
| Completeness | 4.10 /5 |

**Per-question chi tiết:**
| ID | Category | Faithful | Relevant | Recall | Complete |
|----|----------|----------|----------|--------|----------|
| q01 | SLA | 5 | 5 | 5 | 5 |
| q02 | Refund | 5 | 5 | 5 | 5 |
| q03 | Access Control | 5 | 5 | 5 | 5 |
| q04 | Refund | 5 | 5 | 5 | 5 |
| q05 | IT Helpdesk | 5 | 5 | **0** | 5 |
| q06 | SLA | 5 | 5 | 5 | 4 |
| q07 | Access Control | 5 | 5 | 5 | **2** |
| q08 | HR Policy | 5 | 5 | 5 | 5 |
| q09 | Insufficient Context | 5 | **1** | N/A | 3 |
| q10 | Refund | **1** | **1** | 5 | **2** |

**Câu hỏi yếu nhất (điểm thấp):**
- **q05** — context_recall = 0: Source được index dưới tên `it_helpdesk_faq.txt` nhưng expected source là `support/helpdesk-faq.md` → partial string match thất bại do format tên file khác nhau. Retrieval thực tế có tìm được chunk đúng nhưng metric báo 0.
- **q07** — completeness = 2: Model chỉ cite tên cũ "Approval Matrix for System Access", không biết tên mới là "Access Control SOP". Lỗi ở generation — context có thông tin nhưng prompt không ép model so sánh tên cũ/mới.
- **q09** — relevance = 1: Model abstain đúng ("Không đủ dữ liệu") nhưng không đề xuất liên hệ IT Helpdesk. LLM judge chấm relevance thấp vì câu trả lời không hữu ích cho người dùng.
- **q10** — faithfulness = 1, relevance = 1: Model over-abstain — nói "không đủ dữ liệu" dù context retrieve được đủ thông tin về chính sách hoàn tiền tiêu chuẩn (context_recall = 5). Lỗi ở generation: prompt grounding quá chặt khiến model từ chối trả lời câu hỏi suy luận.

**Giả thuyết nguyên nhân (Error Tree):**
- [x] **Retrieval: Dense bỏ lỡ exact keyword / alias** → q07 ("Approval Matrix" là alias tên cũ)
- [x] **Retrieval: Source name mismatch** → q05 (it_helpdesk_faq.txt vs support/helpdesk-faq.md)
- [x] **Generation: Prompt grounding quá chặt** → q10 (over-abstain dù context đủ)
- [x] **Generation: Prompt không ép citation tên mới** → q07 (thiếu tên mới "Access Control SOP")
- [ ] Indexing: Chunking cắt giữa điều khoản
- [ ] Indexing: Metadata thiếu effective_date
- [ ] Retrieval: Top-k quá ít → thiếu evidence

---

## Variant 1 (Sprint 3)

**Ngày:** 2026-04-13  
**Biến thay đổi:** `retrieval_mode` → `"hybrid"` + `use_rerank = True`

**Lý do chọn biến này:**
> Baseline cho thấy 2 failure pattern rõ ràng ở tầng retrieval:
> 1. **q07 (Approval Matrix)**: Query dùng alias tên cũ — dense embedding không gần đủ với tên mới → hybrid (BM25 + dense) sẽ bắt được từ khóa "Approval", "Matrix" nếu chúng xuất hiện trong file.
> 2. **q09 (ERR-403-AUTH)**: Mã lỗi kỹ thuật cụ thể — BM25 mạnh với exact keyword hơn embedding similarity.
> Corpus lẫn lộn ngôn ngữ tự nhiên (policy, SLA) và tên kỹ thuật (error code, SLA label) → hybrid phù hợp.
> Thêm cross-encoder rerank để chọn lại top-3 từ 10 candidate sau khi merge dense+sparse.

**Config thay đổi:**
```
retrieval_mode = "hybrid"   # dense + BM25 với Reciprocal Rank Fusion
use_rerank     = True       # cross-encoder/ms-marco-MiniLM-L-6-v2
# Các tham số còn lại giữ nguyên như baseline
top_k_search   = 10
top_k_select   = 3
```

**Scorecard Variant 1:**
| Metric | Baseline | Variant 1 | Delta |
|--------|----------|-----------|-------|
| Faithfulness | 4.60/5 | 3.40/5 | **-1.20** |
| Answer Relevance | 4.20/5 | 1.00/5 | **-3.20** |
| Context Recall | 4.44/5 | N/A | N/A |
| Completeness | 4.10/5 | 1.00/5 | **-3.10** |

**Nhận xét:**
> Variant 1 **thất bại hoàn toàn** — không phải do thuật toán mà do lỗi môi trường: thư viện `rank_bm25` chưa được cài trong venv. Toàn bộ 10 câu trả về:
> ```
> ERROR: No module named 'rank_bm25'
> ```
> LLM judge chấm faithfulness cao (3.40) vì error message trùng với context (context chứa code snippet đề cập `rank_bm25`). Các metric còn lại đều = 1/5 vì answer không trả lời được câu hỏi nào.
>
> **Câu nào kém hơn baseline:** Tất cả 10 câu — do pipeline crash, không phải do hybrid/rerank tệ hơn dense.

**Kết luận:**
> Variant 1 **không có số liệu có ý nghĩa** để so sánh với baseline. Cần cài `pip install rank-bm25` trước khi chạy lại.
> Giả thuyết ban đầu (hybrid tốt hơn cho alias query và exact keyword) **chưa được kiểm chứng** do lỗi dependency.
> Nếu fix được lỗi này và chạy lại, kỳ vọng:
> - context_recall q07 tăng từ 5 → vẫn 5 (dense đã retrieve được)
> - relevance q09 tăng nhờ BM25 bắt "ERR-403-AUTH"
> - faithfulness q10 cải thiện nếu rerank chọn đúng chunk hơn

---

## Tóm tắt học được

1. **Lỗi phổ biến nhất trong pipeline này là gì?**
   > Generation over-abstain (q10): Model nói "không đủ dữ liệu" dù retrieval đã mang về context đầy đủ. Prompt grounding cần cân bằng giữa "không bịa" và "không từ chối khi có evidence".

2. **Biến nào có tác động lớn nhất tới chất lượng?**
   > Chưa xác định được vì variant bị lỗi dependency. Theo giả thuyết: **prompt engineering** (grounding instruction) có thể fix được q10 mà không cần thay retrieval — chi phí thấp, tác động có thể cao.

3. **Nếu có thêm 1 giờ, nhóm sẽ thử gì tiếp theo?**
   > (1) Fix `rank_bm25` và chạy lại variant để có số liệu thực. (2) Thử Variant 2: chỉ điều chỉnh prompt (thêm điều kiện "nếu context có thông tin liên quan, hãy trả lời dù câu hỏi có vẻ edge case") — đây là biến duy nhất nhắm vào lỗi q10 mà không đụng đến retrieval.
