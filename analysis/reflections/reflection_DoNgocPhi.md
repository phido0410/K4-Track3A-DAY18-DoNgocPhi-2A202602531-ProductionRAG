# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Đỗ Ngọc Phi  
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 04/10/2026  

---

## Phần 1: Mapping bài giảng (Lecture Mapping)
Map từng concept trong lecture vào code thực thi trong lab:

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Sử dụng mô hình sentence-transformers để tính cosine similarity giữa các câu liên tiếp; ngắt chunk khi độ tương đồng nhỏ hơn threshold (0.85). Giúp bảo toàn trọn vẹn mạch ý và ngữ nghĩa của từng đoạn quy định thay vì cắt ngang giữa câu như phương pháp fixed-size cơ bản. |
| Hierarchical chunking | M1 | `chunk_hierarchical()` | Tạo ra cấu trúc parent chunk (kích thước lớn 1200 ký tự) và các child chunk nhỏ hơn (300 ký tự gắn `parent_id`). Khi search, retrieval thực hiện trên child chunk để tăng độ tập trung vector embedding; khi đưa vào context cho LLM thì map ngược lại parent chunk để cung cấp bối cảnh toàn diện, giải quyết vấn đề lost-in-the-middle. |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | Phân tích cú pháp tiêu đề Markdown (H1, H2, H3) và cấu trúc phân cấp tài liệu. Giúp các bảng biểu, danh sách điều khoản trong cùng một mục không bị xé vụn và mang theo header metadata hữu ích để truy vấn. |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | RRF kết hợp xếp hạng từ BM25 (xử lý từ khóa từ vựng chính xác như số tiền, ngày phép, mã văn bản, tên phòng ban) và Dense vector `BAAI/bge-m3` (hiểu ngữ nghĩa trừu tượng). Công thức `1 / (k + rank)` cân bằng tối ưu cả hai nguồn tín hiệu mà không phụ thuộc vào biên độ scale điểm thô. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Mô hình `BAAI/bge-reranker-v2-m3` đọc đồng thời Query và Candidate Chunk qua cơ chế full-attention. Đánh giá lại độ liên quan thực tế và lọc lấy top 3 kết quả chất lượng cao nhất từ top 20 của RRF, triệt tiêu đáng kể các chunk gây nhiễu cho LLM. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Đánh giá toàn diện 4 chỉ số cốt lõi: Faithfulness (độ trung thực không hallucinate), Answer Relevancy (trả lời đúng trọng tâm câu hỏi), Context Precision (độ chính xác của các chunk được retrieve), Context Recall (độ bao phủ đầy đủ thông tin từ context so với ground truth). |
| Contextual embeddings & Enrichment | M5 | `_enrich_single_call()` / `enrich_chunks()` | Sử dụng LLM sinh bản tóm tắt bối cảnh tài liệu (contextual prepend), câu hỏi giả định (HyQA) và trích xuất metadata trong 1 API call duy nhất. Thêm tiền tố context vào đầu chunk giúp vectorizer hiểu được vị trí logic của đoạn trích trong toàn bộ tài liệu gốc. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi kỹ thuật gặp phải (Exact error message):**
  1. `torch freeze / silent hang` trên macOS (Apple Silicon M-series) khi chạy `SentenceTransformer` và `CrossEncoder` với device mặc định `mps` trong các batch kích thước lớn.
  2. `[Errno 8] nodename nor servname provided, or not known thrown while requesting HEAD https://huggingface.co/...` khi `pytest` chạy offline trong môi trường sandbox cô lập mạng.
  3. Điểm đánh giá RAGAS ban đầu trả về 0.0 do thư viện Ragas ngầm thực hiện gọi OpenAI API bất đồng bộ mà cần mạng ngoài (`BypassSandbox`) và yêu cầu biến môi trường `OPENAI_API_KEY`.

- **Nguyên nhân gốc rễ & Cách debug:**
  1. PyTorch Metal Performance Shaders (MPS) trên một số phiên bản gặp race condition / deadlock khi forward pass các transformer model lớn (`bge-m3`, `bge-reranker-v2-m3`) với dynamic sequence length. Cách giải quyết: cấu hình tường minh `device="cpu"` cùng `batch_size=16`. Trên CPU Apple M-series, inference chỉ mất khoảng 20-30s cho toàn bộ corpus, hoàn toàn ổn định và không bao giờ bị treo tiến trình.
  2. Hugging Face `transformers` mặc định luôn cố gắng kết nối mạng để kiểm tra commit hash mới nhất trên HuggingFace Hub. Cách giải quyết: bổ sung `os.environ["HF_HUB_OFFLINE"] = "1"` và `os.environ["TRANSFORMERS_OFFLINE"] = "1"` vào các hàm tải model, tải thẳng từ thư mục cache local `~/.cache/huggingface/hub/` mà không gửi request check hash lên server.
  3. Cấp quyền mạng khi chạy pipeline tổng hợp `main.py` và truyền rõ `api_key` cho client `OpenAI`, kết quả Production RAG vượt mốc tiêu chuẩn với Faithfulness 0.8192 và Relevancy 0.8201.

- **Kiến thức còn thiếu & Cách khắc phục:**
  - Hiểu sâu hơn về cách RAGAS tính toán điểm số: RAGAS sử dụng chính LLM làm Judge để phân tích câu. Nếu câu trả lời có tính toán suy luận (reasoning) mà trong context chỉ có công thức thì chỉ số Faithfulness vẫn có thể bị chấm thấp nếu không giải trình rõ ràng. Khắc phục bằng cách yêu cầu Generator nêu rõ nguồn công thức từ điều khoản tài liệu trước khi đưa ra kết quả tính.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

Dựa trên những kỹ thuật đã học và thực hành, lập kế hoạch cụ thể áp dụng vào project cá nhân:

### Project: Trợ lý AI Hỏi đáp Quy trình Vận hành Doanh nghiệp (Enterprise SOP Assistant)

#### 1. Hiện trạng
- **Pipeline hiện tại:** Sử dụng LangChain cơ bản với RecursiveCharacterTextSplitter (chunk_size=1000, overlap=100), lưu trữ trên ChromaDB và gọi trực tiếp `gpt-4o-mini` để sinh phản hồi.
- **Vấn đề / Bottlenecks đang gặp:**
  - Precision thấp khi hỏi về các điều khoản có chứa các con số cụ thể (mức thưởng, thời hạn hợp đồng, số ngày phép).
  - Thiếu khả năng phân biệt phiên bản chính sách cũ và mới khi quy định được cập nhật hàng năm.
  - Không có framework đánh giá định lượng, chủ yếu kiểm tra thủ công bằng mắt.

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:** Chuyển sang kết hợp **Structure-aware** và **Hierarchical chunking**. Các file quy trình SOP được phân tách theo Heading (H1, H2, H3) và tạo Parent-Child chunks để giữ nguyên bảng biểu quy định.
2. **Search retrieval:** Áp dụng **Hybrid Search (BM25 + Dense BAAI/bge-m3)** kết hợp thuật toán **Reciprocal Rank Fusion (RRF)**. BM25 giải quyết triệt để vấn đề tìm kiếm các thuật ngữ mã hóa và số hiệu văn bản; Dense vector tìm kiếm theo ngữ cảnh người dùng.
3. **Reranking:** Triển khai **Cross-Encoder Reranker (`bge-reranker-v2-m3`)** với top 15 candidate lọc xuống top 3. Cắt bỏ hoàn toàn các đoạn văn rác ngoài lề.
4. **Evaluation:** Thiết lập bộ benchmark tự động với **RAGAS 4 metrics (Faithfulness, Answer Relevancy, Context Precision, Context Recall)** chạy định kỳ qua CI/CD mỗi khi cập nhật cơ sở dữ liệu SOP.
5. **Enrichment:** Sử dụng **Contextual Embeddings** (tóm tắt bối cảnh tài liệu vào đầu mỗi chunk) trong bước indexing offline để cải thiện khả năng định vị tài liệu.

#### 3. Timeline triển khai
- **Tuần 1:**
  - Tái cấu trúc cơ sở tri thức SOP theo Markdown chuẩn hoá.
  - Implement module Structure-aware & Hierarchical Chunking.
  - Thiết lập Hybrid Search (Qdrant + BM25) và kiểm thử RRF.
- **Tuần 2:**
  - Tích hợp Cross-Encoder Reranker và kiểm soát latency dưới 500ms.
  - Tích hợp pipeline đánh giá RAGAS trên tập 50 câu hỏi vàng (Golden Test Set).
  - Tinh chỉnh System Prompt xử lý xung đột phiên bản chính sách và deploy ứng dụng.

