# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Đỗ Ngọc Phi  
**Khóa:** K4 - Track 3A  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.7767 | 0.8192 | +0.0425 |
| Answer Relevancy | 0.7677 | 0.8201 | +0.0524 |
| Context Precision | 0.9250 | 0.9250 | 0.0000 |
| Context Recall | 0.9250 | 0.8500 | -0.0750 |

---

## Bottom-5 Failures

### #1
- **Question:** Khi phát hiện malware trên máy, nhân viên có nên tự xử lý không?
- **Expected:** KHÔNG. Nhân viên tuyệt đối không được tự ý xử lý malware. Phải báo cáo trong vòng 1 giờ qua helpdesk@cty.vn hoặc hotline CNTT. Tự ý xử lý bị coi là vi phạm nghiêm trọng.
- **Got:** Nhân viên **không nên tự xử lý** malware trên máy mà phải chờ sự hướng dẫn của đội CNTT. Việc tự ý xử lý sẽ bị xem là vi phạm nghiêm trọng quy chế bảo mật.
- **Worst metric:** answer_relevancy (0.00)
- **Error Tree:** Output đúng ý chính nhưng thiếu chi tiết hành động tức thời (báo cáo trong 1h, hotline/email) → Context đúng và đủ (có quy định bảo mật) → Query embedding tương thích tốt → LLM judge của RAGAS đánh giá câu trả lời ngắn chưa bao hàm đủ semantic keywords so với câu hỏi mở.
- **Root cause:** Prompt sinh câu trả lời cô đọng khiến LLM bỏ sót hành động phản ứng sự cố cụ thể (báo cáo trong vòng 1 giờ, email/hotline tiếp nhận).
- **Suggested fix:** Cải tiến System Prompt của Generator để khi trả lời các câu hỏi về quy trình/bảo mật, bắt buộc phải nêu rõ: hành vi bị cấm, thời hạn báo cáo và kênh tiếp nhận hỗ trợ.

---

### #2
- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** Nghỉ 16-30 ngày cần phê duyệt của Giám đốc điều hành (CEO). Lưu ý: nghỉ trên 14 ngày không lương, nhân viên phải tự đóng phần bảo hiểm của mình.
- **Got:** Nghỉ phép không lương 20 ngày cần phê duyệt của Giám đốc điều hành (CEO).
- **Worst metric:** faithfulness (0.50)
- **Error Tree:** Output đúng thẩm quyền phê duyệt nhưng thiếu điều kiện phụ về bảo hiểm xã hội → Context chứa cả bảng phân cấp nghỉ phép và lưu ý đóng bảo hiểm → LLM tóm tắt lược bỏ điều khoản phát sinh khi nghỉ trên 14 ngày.
- **Root cause:** Câu trả lời chỉ tập trung trực diện vào đối tượng phê duyệt ("cần ai phê duyệt") mà không trích dẫn hệ quả đi kèm được ghi chú trong cùng điều khoản chính sách.
- **Suggested fix:** Tinh chỉnh prompt: "Khi giải thích điều kiện nghỉ phép hoặc chính sách nhân sự, luôn trích dẫn kèm theo các điều kiện ràng buộc hoặc chi phí/nghĩa vụ liên quan".

---

### #3
- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Laptop 30 triệu nằm trong khoảng 5-50 triệu nên cần Giám đốc phòng ban (Director) phê duyệt. Ngoài ra, mua sắm thiết bị CNTT cần có xác nhận cấu hình kỹ thuật từ phòng CNTT trước khi đề xuất. Cần đính kèm ít nhất 3 báo giá vì trên 10 triệu.
- **Got:** Để mua chiếc laptop 30 triệu, cần có phê duyệt của Kế toán trưởng (vì giá trị đơn hàng từ 5.000.000 VNĐ trở lên). Ngoài ra, trước khi đề xuất, cần có xác nhận của phòng CNTT về cấu hình kỹ thuật của laptop.
- **Worst metric:** context_recall (0.33)
- **Error Tree:** Output nhầm lẫn cấp phê duyệt giữa Kế toán trưởng và Giám đốc bộ phận, thiếu yêu cầu 3 báo giá → Context retrieved bị thiếu chunk phân cấp thẩm quyền tài chính mua sắm tổng thể → Top 3 sau Reranking ưu tiên chunk quy trình CNTT hơn chunk ma trận thẩm quyền tài chính.
- **Root cause:** Query dạng đa ý ("ai phê duyệt VÀ cần gì từ phòng CNTT") đòi hỏi ghép thông tin từ 2 văn bản khác nhau (Quy trình mua sắm và Quy định CNTT), retrieval chỉ lấy được 1 trong 2 văn bản vào top đầu.
- **Suggested fix:** Áp dụng Multi-Query Expansion hoặc Sub-query Decomposition: tách câu hỏi phức hợp thành 2 câu truy vấn con ("Thẩm quyền phê duyệt mua sắm 30 triệu" và "Yêu cầu phòng CNTT khi mua laptop") để retrieve đủ các văn bản liên quan.

---

### #4
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Thời hạn thanh toán là 15 ngày. Quá hạn 5 ngày, bị tính phí 2%/tháng trên 15.000.000 VNĐ = 300.000 VNĐ/tháng (tính pro-rata khoảng 50.000 VNĐ cho 5 ngày).
- **Got:** Nhân viên tạm ứng 15 triệu và thanh toán sau 20 ngày, tức là quá hạn 5 ngày. Phạt quá hạn được tính là 2%/tháng trên số tiền chưa hoàn ứng. Tính toán cụ thể: Phạt mỗi tháng: 2% của 15 triệu = 300.000 VNĐ; Thời gian quá hạn: 5 ngày tương đương với khoảng 1/6 tháng; Phạt thực tế: 300.000 VNĐ * (5/30) = 50.000 VNĐ.
- **Worst metric:** faithfulness (0.33)
- **Error Tree:** Output tính toán số học rất chính xác và khớp ground truth → RAGAS Faithfulness judge đối chiếu context: Context gốc chỉ ghi công thức "phạt 2%/tháng tính theo ngày quá hạn" mà không có sẵn con số 50.000 VNĐ → LLM judge coi phép tính suy luận số học là suy diễn (hallucination) không có trực tiếp trong text.
- **Root cause:** Đặc tính khắt khe của metric Faithfulness trong RAGAS: không phân biệt giữa hallucination sai sự thật và reasoning số học hợp lệ dựa trên quy định có sẵn trong văn bản.
- **Suggested fix:** Thêm chỉ dẫn giải thích rõ ràng phép toán trích xuất từ văn bản ("Dựa theo công thức X trong điều Y, số tiền tính ra là...") hoặc cấu hình Few-shot prompt cho evaluator hiểu reasoning hợp lệ.

---

### #5
- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** Theo chính sách hiện hành (v2.0), mật khẩu phải được thay đổi mỗi 120 ngày. Chính sách cũ yêu cầu 90 ngày nhưng đã bị thay thế.
- **Got:** Theo chính sách hiện hành, mật khẩu phải được thay đổi mỗi 120 ngày.
- **Worst metric:** faithfulness (0.33)
- **Error Tree:** Output chọn đúng chính sách mới nhất (120 ngày) → Context retrieved chứa cả văn bản v1.0 (90 ngày) và v2.0 (120 ngày) → Generator không nhắc tới việc "văn bản cũ là 90 ngày đã bị thay thế" khiến câu trả lời bị RAGAS chấm điểm thấp do không giải trình trọn vẹn sự xung đột trong context.
- **Root cause:** Khi có tài liệu mâu thuẫn phiên bản trong context, mô hình đã lọc đúng phiên bản mới nhưng không tuyên bố loại trừ phiên bản cũ.
- **Suggested fix:** Cải tiến prompt quy định: "Nếu phát hiện nhiều phiên bản chính sách khác nhau trong context, phải nêu rõ phiên bản đang áp dụng và xác nhận phiên bản cũ đã hết hiệu lực".

---

## Case Study (cho presentation)

**Question chọn phân tích:**
"Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?"

**Error Tree walkthrough:**
1. **Output đúng?** → Sai một phần. Trả lời được yêu cầu cấu hình từ phòng CNTT nhưng nhầm cấp phê duyệt (nêu Kế toán trưởng thay vì Giám đốc phòng ban) và thiếu yêu cầu 3 báo giá.
2. **Context đúng?** → Chưa đủ. Reranker lấy được chunk từ quy chế CNTT và chunk tạm ứng/chi tiêu chung, nhưng chunk về Bảng ma trận thẩm quyền phê duyệt mua sắm tài sản cố định (phân cấp 5-50 triệu thuộc Director) bị rơi khỏi top 3.
3. **Query rewrite OK?** → Câu hỏi là câu ghép 2 ý khác nhau. Tìm kiếm dense + lexical theo query nguyên bản bị pha loãng giữa từ khóa "laptop/CNTT" và "30 triệu/phê duyệt".
4. **Fix ở bước:** Bước **Query Transformation (Decomposition)** trước retrieval và tăng `top_k` của Reranker lên 5 hoặc sử dụng Metadata filtering theo loại tài liệu (`source=quy_che_tai_chinh.md` kết hợp `source=chinh_sach_cntt.md`).

**Nếu có thêm 1 giờ, sẽ optimize:**
- Tích hợp kỹ thuật **Sub-Query Decomposition** để tự động tách các câu hỏi đa ý thành nhiều query con độc lập trước khi gửi tới Hybrid Search.
- Nâng cấp **Hierarchical Parent Retrieval**: trả về ngữ cảnh parent chunk bao hàm toàn bộ bảng phân cấp phê duyệt tài chính thay vì chỉ lấy child chunk cục bộ.
