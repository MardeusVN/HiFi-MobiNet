# Bảng ablation hợp nhất: từ HiFi-GAN baseline đến MRF(1,1,1)

**Mục đích:** gộp toàn bộ các bước quyết định kiến trúc thành 1 bảng duy
nhất, mỗi dòng cho thấy đóng góp riêng của từng thay đổi, thay vì rải rác
trong nhiều báo cáo (`mbconv_resblock_report.md`,
`mrf_full_investigation_report.md`, `final_4way_comparison_report.md`).

Cột "loss_mel" ở các dòng ablation sớm (overfit-test 800 bước, chưa train
đầy đủ) dùng để SO SÁNH TƯƠNG ĐỐI giữa các lựa chọn kiến trúc lúc khảo sát —
không phải kết quả cuối. Các dòng cuối (train 1500 epoch thật) mới là số
liệu final.

---

## Giai đoạn 1 — Chọn loại khối thay resblock (overfit-test 800 bước, tuần tự)

| # | Thay đổi | Params | loss_mel | Ghi chú |
|---|---|---:|---:|---|
| 1a | Baseline (resblock="2", dense dilated-conv) | 2.240.000 | 10.5092 | Mốc gốc |
| 1b | +Khối inverted-residual (kernel=7, expansion=2) | ~2.14M | 8.91–9.25 | Thắng sau khi sửa lỗi zero-init (xem §1c) |
| 1c | **Fix: zero-init `pw_project`** (khối bắt đầu là identity no-op) | — | — | **Root-cause fix cho bất ổn training** — không phải giảm LR như nghi ngờ ban đầu |
| 1d | +MobileNetV2 defaults thật (expansion=6, kernel=3) | 2.139.712 | tốt hơn 1b | Đổi kernel 7→3, expansion 2→6 |

## Giai đoạn 2 — Per-stage expansion schedule (giải quyết RTF 3.7x chậm ở expansion đồng nhất)

| # | Expansion (tầng 1,2,3) | Params | loss_mel | Tốc độ (so baseline) |
|---|---|---:|---:|---|
| 2a | (6,6,6) đồng nhất | 2.139.712 | — | **chậm hơn baseline ~3x** (phát hiện lỗ hổng phương pháp luận: chưa từng đo RTF trước đó) |
| 2b | (6,6,1) | — | 9.3273 | vẫn chậm |
| 2c | (4,4,1) | 1.943.424 | 9.3789 | nhanh hơn baseline nhẹ |
| 2d | (2,2,1) | 1.771.136 | 9.0462 | nhanh hơn |
| 2e | **(1,1,1)** | **1.684.992** | **8.9742 (tốt nhất tuần tự)** | **nhanh nhất, nhẹ nhất** |

## Giai đoạn 3 — Sequential vs. MRF song song (đúng tinh thần HiFi-GAN gốc)

| # | Cơ chế | Expansion | Params | loss_mel | Tốc độ 12 luồng (PyTorch) |
|---|---|---|---:|---:|---|
| 3a | Tuần tự (2 khối nối tiếp) | (1,1,1) | 1.684.992 | 8.9742 | 8.04ms |
| 3b | **MRF song song (3 nhánh, kernel 3/5/7)** | (1,1,1) | 1.732.256 | 9.0788 | 17.30ms (chậm hơn tuần tự nhưng vẫn nhanh hơn baseline 19.94ms) |
| 3c | MRF song song | (4,4,1) | 2.123.360 | 8.9734 | 19.53ms (~hòa baseline, mất lợi thế tốc độ) |

→ **Chọn 3b (MRF song song, expansion=1,1,1)**: giữ đúng tinh thần thiết kế HiFi-GAN (multi-receptive-field thật), vẫn nhanh hơn baseline, loss_mel gần sát tuần tự (~1.2%).

## Giai đoạn 4 — Training thật đầy đủ 1500 epoch (kết quả cuối, seed cố định, test-set đúng)

| # | Model | val_loss_mel | RTF (PyTorch, đo sạch) | UTMOSv2 | WER | Params | GFLOPs/s |
|---|---|---:|---:|---:|---:|---:|---:|
| 4a | Baseline (resblock="2"), epoch=1489/1500 | 19.7882 | 0.0434 | 3.125 | 0.1230 | 1.662.976 | 3.9016 |
| 4b | **MRF(1,1,1), epoch=1386/1500** | **18.7681** | **0.0370** | **3.651** | **0.0983** | **1.154.560** | **1.2306** |

→ Ở PyTorch/FP32: MRF thắng cả 4 trục (nhanh hơn ~14.7%, UTMOSv2 tốt hơn ~16.8%, WER tốt hơn ~20.1%, nhẹ hơn ~30.6% tham số, ít FLOPs hơn ~68.5%).

## Giai đoạn 5 — Quantization (PTQ) + đo lại đúng backend/phần cứng (ONNX Runtime, VNNI, 2 luồng)

| # | Model | RTF | UTMOSv2 | WER | Ý nghĩa thống kê (so baseline) |
|---|---|---:|---:|---:|---|
| 5a | Baseline FP32 (ONNX) | 0.0388 | 3.132 | 0.1230 | mốc |
| 5b | Baseline PTQ INT8 (flow+enc_p+dp, dec FP32) | 0.0345 | 3.121 | 0.1135 | mốc |
| 5c | MRF FP32 (ONNX) | 0.0416 | 3.537 | 0.1043 | RTF p<0.001 (MRF chậm hơn); UTMOSv2 p<0.001 (MRF tốt hơn); WER p=0.012 (MRF tốt hơn, yếu) |
| 5d | MRF PTQ INT8 (flow+enc_p+dp, dec FP32) | 0.0371 | 3.490 | 0.1124 | RTF p<0.001 (MRF chậm hơn); UTMOSv2 p<0.001 (MRF tốt hơn); **WER p=0.61 (KHÔNG có ý nghĩa)** |

→ **Phát hiện quan trọng, đảo ngược 1 phần kết luận giai đoạn 4**: khi đo qua ONNX Runtime (backend triển khai thật) thay vì PyTorch, **baseline nhanh hơn MRF** (không phải ngược lại) — do depthwise conv trong MRF bị giới hạn băng thông bộ nhớ, không tận dụng đa luồng hiệu quả bằng dense conv, đặc biệt rõ ở câu dài (>5s: MRF chậm hơn ~8-9%). Xem `docs/mrf_full_investigation_report.md` và hội thoại liên quan để có toàn bộ chuỗi kiểm chứng (test dec-only cô lập, T-sweep, xác nhận qua cả PyTorch lẫn ONNX Runtime).

---

## Tổng kết đóng góp từng thay đổi

| Thay đổi tích lũy | Tác động chính |
|---|---|
| Dense conv → inverted-residual (MobileNetV2-style) | Giảm tham số đáng kể, cần zero-init để ổn định training |
| Expansion đồng nhất → per-stage (X,X,1) | Sửa lỗi RTF chậm 3x ở PyTorch |
| Tuần tự → MRF song song | Giữ đúng tinh thần HiFi-GAN, đánh đổi tốc độ nhỏ lấy tính trung thực kiến trúc |
| Train đủ 1500 epoch (không dừng sớm) | Biên độ thắng lớn nhất so với các mốc train ngắn hơn |
| Đo qua ONNX Runtime thay vì chỉ PyTorch | **Đảo ngược kết luận RTF** — bài học phương pháp luận quan trọng nhất của toàn bộ nghiên cứu |
| PTQ INT8 (flow+enc_p+dp) | Giảm size thêm, giữ hầu hết lợi thế UTMOSv2 của MRF, nhưng làm mất ý nghĩa thống kê của lợi thế WER |
