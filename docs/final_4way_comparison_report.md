# So sánh RTF / WER / UTMOSv2: baseline vs MRF(1,1,1), FP32 và PTQ INT8

**Phạm vi:** báo cáo so sánh tập trung 4 cấu hình cuối cùng — HiFi-GAN
baseline (FP32 và PTQ INT8), và MRF(1,1,1) train đủ 1500 epoch (FP32 và
PTQ INT8). **Hướng QAT đã thử nhưng chủ động dừng** (xem `docs/TODO.md` và
`paper_ready_report.md` §3.2) — không có INT8 thật tăng tốc do giới hạn
kiến trúc wrapper, chi phí sửa+train lại không tương xứng khi PTQ đã cho
kết quả tốt. Toàn bộ số RTF trong báo cáo này đo qua **ONNX Runtime CPU EP
(có AVX-VNNI), 2 luồng cố định** — cùng 1 backend cho cả 4 dòng, tránh lỗi
trộn backend (PyTorch vs ONNX) đã phát hiện và sửa giữa chừng. Seed cố định
(`torch.manual_seed(1234)`) và mỗi model dùng đúng test-set 500 câu riêng
(tránh train/test leakage — xem `mrf_full_investigation_report.md` §8).

---

## 1. Bảng so sánh

| Model | Checkpoint | RTF (ONNX/VNNI, 2 luồng) ↓ | UTMOSv2 ↑ | WER ↓ | Generator params | Generator size (FP32) | GFLOPs/s audio | Pipeline ONNX (toàn bộ) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Baseline FP32** | epoch=1489/1500 (99%, hội tụ đầy đủ) | 0.0388 | 3.132 | 0.1230 | 1.662.976 | 6.35 MB | 3.9016 | 65.4 MB |
| **Baseline PTQ INT8** | flow+enc_p+dp int8, dec FP32 | **0.0345** | 3.121 | 0.1135 | 1.662.976 (dec không đổi) | 6.35 MB (dec không đổi) | — | 23.0 MB |
| **MRF(1,1,1) FP32** | epoch=1386/1500 (92%, train đủ lịch) | 0.0416 | **3.537** | **0.1043** | **1.154.560** | **4.41 MB** | **1.2306** | 63.5 MB |
| **MRF(1,1,1) PTQ INT8** | flow+enc_p+dp int8, dec FP32 | 0.0371 | **3.490** | 0.1124 | 1.154.560 (dec không đổi) | 4.41 MB (dec không đổi) | — | **21.1 MB** |

**Ghi chú kích thước**: "Generator size" chỉ tính riêng vocoder (`dec`) —
phần duy nhất khác nhau giữa 2 kiến trúc; không đổi giữa FP32 và PTQ vì
`dec` không quantize được (xem §3.4). "GFLOPs/s audio" đo thủ công qua
forward-hook trên Conv1d/ConvTranspose1d (MACs chuẩn ×2=FLOPs), chỉ tính
cho `dec` — MRF nhẹ hơn baseline **30.6%** tham số, ít hơn **68.5%**
FLOPs. "Pipeline ONNX" là kích thước file export đầy đủ (TextEncoder +
Flow + DurationPredictor + Generator), ở hàng PTQ là sau khi `flow/enc_p/dp`
đã lượng tử hoá INT8 (`dec` vẫn FP32).

---

## 2. Kiểm định thống kê (Mann-Whitney U, 2 mẫu độc lập vì 2 test-set khác nhau)

| Metric | Điều kiện | p-value | Ý nghĩa | 95% CI chênh lệch (MRF − baseline) |
|---|---|---|---|---|
| RTF | FP32 | 9.6×10⁻⁹⁴ | *** | [+0.0026, +0.0030] — MRF chậm hơn |
| UTMOSv2 | FP32 | 3.2×10⁻⁹³ | *** | [+0.373, +0.436] — MRF tốt hơn |
| WER | FP32 | 0.012 | * (yếu) | [−0.0348, −0.0025] — MRF tốt hơn |
| RTF | PTQ INT8 | 7.7×10⁻⁶⁸ | *** | [+0.0023, +0.0028] — MRF chậm hơn |
| UTMOSv2 | PTQ INT8 | 1.0×10⁻⁸⁵ | *** | [+0.339, +0.400] — MRF tốt hơn |
| **WER** | **PTQ INT8** | **0.607** | **không có ý nghĩa** | [−0.0184, +0.0169] — không phân biệt được |

---

## 3. So sánh theo cặp

### 3.1. FP32: MRF thắng UTMOSv2 và WER (có ý nghĩa thống kê), nhưng chậm hơn RTF (cũng có ý nghĩa thống kê)

| | Baseline FP32 | MRF FP32 | Chênh lệch |
|---|---:|---:|---|
| RTF ↓ | 0.0388 | 0.0416 | **MRF chậm hơn ~7.2%** (p<0.001) |
| UTMOSv2 ↑ | 3.132 | 3.537 | **MRF tốt hơn ~12.9%** (p<0.001) |
| WER ↓ | 0.1230 | 0.1043 | **MRF tốt hơn ~15.2%** (p=0.012) |

### 3.2. PTQ INT8: cùng pattern, WER không còn phân biệt được

| | Baseline INT8 | MRF INT8 | Chênh lệch |
|---|---:|---:|---|
| RTF ↓ | 0.0345 | 0.0371 | **MRF chậm hơn ~7.5%** (p<0.001) |
| UTMOSv2 ↑ | 3.121 | 3.490 | **MRF tốt hơn ~11.8%** (p<0.001) |
| WER ↓ | 0.1135 | 0.1124 | ~ngang nhau (p=0.607, **không có ý nghĩa**) |

### 3.3. Vì sao MRF chậm hơn ở RTF dù ít FLOPs hơn nhiều (68.5%)

Depthwise conv trong `ResBlockInverted` bị giới hạn băng thông bộ nhớ
(memory-bound), không tận dụng đa luồng hiệu quả như dense conv của
baseline — hiệu ứng này **tăng dần theo độ dài câu** (T-sweep: chênh lệch
RTF từ +3.5% ở audio ngắn lên +14.3% ở audio dài ~8s), đã kiểm chứng chéo
qua 3 cách độc lập (test dec-only cô lập, T-sweep tổng hợp, phân tích lại
500 mẫu thật theo nhóm độ dài) và cả 2 backend (PyTorch lẫn ONNX Runtime,
chỉ khác ngưỡng T xảy ra). Chi tiết đầy đủ: `paper_ready_report.md` §2.4.

**Phát hiện mới (xem `docs/TODO.md` mục 1)**: benchmark nhanh (chưa train)
cho thấy bản **resblock tuần tự** (không phải MRF song song) của cùng khối
`ResBlockInverted` **không hề bị đảo chiều RTF** ở bất kỳ T nào đã test —
nhiều khả năng vấn đề đến từ việc MRF song song nhân 3 nhánh depthwise
cùng lúc, không phải bản thân khối. Đưa vào TODO để train lại và xác nhận
đầy đủ.

---

## 4. Kết luận

- **Nếu triển khai FP32**: MRF(1,1,1) thắng chất lượng (UTMOSv2, WER) có ý
  nghĩa thống kê và nhẹ hơn/ít FLOPs hơn nhiều, nhưng **chậm hơn baseline
  ~7% khi đo qua runtime triển khai thật (ONNX Runtime)** — ngược với kết
  quả đo qua PyTorch (nhanh hơn ~14.7%). Đây là điểm cần cân nhắc kỹ tuỳ
  use-case (độ dài audio trung bình, ràng buộc latency).
- **Nếu triển khai PTQ INT8**: cùng pattern — MRF vẫn thắng UTMOSv2 rõ,
  nhưng WER giữa 2 model **không còn phân biệt được về thống kê**, và MRF
  vẫn chậm hơn baseline ~7.5% qua ONNX Runtime.
- **`dec` (vocoder) của MRF không quantize được qua PTQ** (lỗi ONNX
  Runtime với depthwise conv) — chỉ `flow+enc_p+dp` (dùng chung, không
  phải phần novelty) quantize được ở cả 2 model.
- Hướng khắc phục RTF khả thi nhất (chuyển sang resblock tuần tự) đã có
  tín hiệu tích cực ban đầu, ghi nhận ở `docs/TODO.md` để làm sau.
