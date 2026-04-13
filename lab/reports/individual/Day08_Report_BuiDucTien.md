# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Bùi Đức Tiến  
**Vai trò trong nhóm:** Documentation Owner  
**Ngày nộp:** 2026-04-13  
**Độ dài yêu cầu:** 500–800 từ

---

## 1. Tôi đã làm gì trong lab này? (100-150 từ)

Trong lab này, tôi đảm nhận vai trò Documentation Owner — chịu trách nhiệm ghi lại và trình bày toàn bộ quyết định kỹ thuật của nhóm dưới dạng văn bản có thể kiểm chứng.

Cụ thể, tôi hoàn thiện `docs/architecture.md`: điền thông số chunking thực tế (chunk_size, overlap, embedding model), mô tả retrieval config của baseline và variant, hoàn chỉnh sơ đồ Mermaid cho pipeline end-to-end. Song song, tôi duy trì `docs/tuning-log.md`: ghi nhận error tree từ kết quả baseline, ghi rõ lý do chọn một biến để tune (hybrid retrieval vì corpus lẫn lộn ngôn ngữ tự nhiên và tên kỹ thuật), điền bảng so sánh A/B sau khi nhóm chạy xong eval.

Công việc của tôi phụ thuộc vào output của cả ba Sprint còn lại — tôi không thể điền số liệu vào tuning-log nếu Eval Owner chưa có scorecard, và không thể viết retrieval config nếu Tech Lead chưa confirm variant cuối. Ngược lại, architecture.md là tài liệu tham chiếu để các thành viên khác hiểu nhau khi review code.

---

## 2. Điều tôi hiểu rõ hơn sau lab này (100-150 từ)

Trước lab, tôi nghĩ "document" chỉ là viết lại những gì đã làm sau khi xong. Sau lab này, tôi hiểu documentation trong pipeline ML phải đi song song với code, không phải sau.

Cụ thể, khi điền `tuning-log.md`, tôi phải hiểu rõ **error tree**: một điểm số thấp trong scorecard có thể bắt nguồn từ indexing (chunk cắt giữa điều khoản), retrieval (dense bỏ lỡ alias), hoặc generation (prompt không đủ grounding). Nếu không phân loại được failure mode, việc chọn biến để tune sẽ là đoán mò. Đây là lý do tại sao A/B rule quan trọng: khi chỉ đổi một biến, nếu điểm tăng thì biết chính xác biến đó có tác dụng; nếu đổi đồng thời 3 biến và điểm tăng, vẫn không biết nguyên nhân thực sự.

Viết `architecture.md` cũng giúp tôi nhận ra rằng một quyết định tưởng nhỏ như chunk_size ảnh hưởng đến cả retrieval lẫn generation — chunk quá nhỏ làm mất context, quá lớn làm LLM bị "lost in the middle".

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn (100-150 từ)

Khó khăn lớn nhất là điền `tuning-log.md` khi chưa có số liệu scorecard — tôi phải viết phần "giả thuyết nguyên nhân" trước khi chạy eval, dựa vào phân tích thủ công trên một vài câu hỏi mẫu. Điều này khó hơn tôi tưởng vì phải vừa hiểu codebase (index.py, rag_answer.py) vừa suy luận về failure mode mà không có số liệu định lượng.

Điều ngạc nhiên: khi đọc code `score_context_recall()` trong `eval.py`, tôi thấy metric này không dùng LLM mà chỉ dùng partial string match theo tên file. Điều này có nghĩa là context recall có thể cho false negative nếu tên file trong metadata và trong `expected_sources` có format khác nhau — ví dụ `support/sla-p1-2026.pdf` vs `sla_p1_2026`. Tôi đã ghi nhận điều này vào tuning-log như một điểm cần lưu ý khi đọc kết quả scorecard, để nhóm không hiểu nhầm context recall thấp là do retrieval kém.

---

## 4. Phân tích một câu hỏi trong scorecard (150-200 từ)

**Câu hỏi:** `gq07` — *"Mức phạt vi phạm SLA P1 là bao nhiêu?"*

**Phân tích:**

Đây là câu **abstain question** — thông tin về mức phạt vi phạm SLA không có trong bất kỳ tài liệu nào được index. Pipeline cần nhận ra điều đó và từ chối trả lời thay vì bịa ra con số.

Ở **Baseline (dense)**, dense retrieval sẽ tìm các chunk liên quan đến SLA P1 vì embedding của "mức phạt vi phạm SLA P1" gần với các đoạn mô tả SLA. Nguy cơ: pipeline retrieve được chunk có nội dung SLA (response time, escalation) nhưng **không có** thông tin về penalty. Nếu grounded prompt không đủ chặt, LLM có thể suy diễn ra con số từ general knowledge — đây là hallucination, dẫn đến penalty −5 điểm theo rubric.

Câu trả lời đúng phải là dạng: *"Không đủ dữ liệu trong tài liệu để trả lời câu hỏi về mức phạt vi phạm SLA."*

Ở **Variant (hybrid + rerank)**, vấn đề tương tự — lỗi nằm ở **generation**, không phải retrieval. Hybrid hay rerank không giúp được nếu LLM không được nhắc đủ rõ rằng thiếu context thì phải abstain. Đây là lý do câu `build_grounded_prompt()` trong code có dòng: *"If the context does not contain enough information... say explicitly: 'Không đủ dữ liệu...'"*

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì? (50-100 từ)

**Thứ nhất:** Bổ sung vào `tuning-log.md` một cột **Failure Mode** trong bảng per-question, không chỉ ghi điểm. Kết quả scorecard hiện tại chỉ cho thấy điểm số — không giúp nhóm sprint tiếp theo biết cần fix ở tầng nào.

**Thứ hai:** Viết một script nhỏ tự động sinh diff giữa `scorecard_baseline.md` và `scorecard_variant.md` và append vào `tuning-log.md`. Hiện tại việc này làm tay dễ sai số, đặc biệt khi cần so sánh từng câu hỏi theo từng metric.

---

*File: `reports/individual/Day08_Report_BuiDucTien.md`*
