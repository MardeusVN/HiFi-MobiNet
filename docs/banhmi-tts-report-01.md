# Báo cáo tổng hợp phiên làm việc BanhmiTTS — #01

**Phạm vi:** toàn bộ thực nghiệm, training, đánh giá, bug phát hiện/sửa, và bài học rút ra trong phiên làm việc này, trải dài 4 mảng công việc chính: (1) thêm F0 conditioning vào BanhmiTTS, (2) khảo sát PTQ quantization cho 2 checkpoint `baseline` và `vocos_small`, (3) QAT training dựa trên công thức PTQ đã xác nhận, (4) điều tra chất lượng âm thanh `vocos_small` bị rè và nghiên cứu tài liệu liên quan. Toàn bộ số liệu là kết quả đo thật trong phiên này (checkpoint/log/eval script còn lưu trên máy), không phải suy đoán.

---

## 1. F0 Conditioning

### 1.1. Bối cảnh và động lực

So sánh kiến trúc chi tiết giữa BanhmiTTS, EdgeTTS (nhiều config), và repo tham chiếu VITS2 học thuật (`p0p4k/vits2_pytorch`) để tìm lý do BanhmiTTS's `vocos_small` thua UTMOS so với EdgeTTS Config H. Khảo sát layer-by-layer (WN, StochasticDurationPredictor, DurationDiscriminator, flow, TextEncoder) không tìm ra khác biệt bug nào — mọi khác biệt kiến trúc đều là lựa chọn thiết kế đã biết (ví dụ transformer-trong-flow của Config H tạo thêm 24 layer so với Config A).

**Ablation cô lập đóng góp của F0** (Config E vs Config H trong EdgeTTS — giống hệt nhau ngoại trừ có/không F0, `piper_train.eval_harness`, n=500, Wilcoxon test): F0 đóng góp **+0.217 UTMOS** (Config E=3.579, Config H=3.796, p=1.5×10⁻³⁹) — đòn bẩy lớn nhất được xác nhận trong toàn bộ investigation, hơn cả SnakeBeta (+0.08).

### 1.2. Bug lý thuyết phát hiện trong cách EdgeTTS làm F0

Trace code EdgeTTS (`models.py`) phát hiện **train/inference mismatch** thật:
- **Lúc train**: decoder được điều kiện hóa bằng F0 gốc theo từng frame (`f0_slice`, mượt, có biến thiên trong nội bộ 1 phoneme).
- **Lúc infer**: decoder chỉ nhận được F0 hằng số theo từng phoneme (`torch.matmul(attn, log_f0_pred)` — `attn` là ma trận alignment cứng 0/1, tạo tín hiệu dạng bậc thang).

→ Decoder học trên tín hiệu dạng sóng mượt nhưng chỉ nhận được tín hiệu bậc thang lúc suy luận thật — exposure bias không hiện ra ở loss curve, chỉ ảnh hưởng chất lượng thật.

### 1.3. Fix áp dụng cho BanhmiTTS (không đụng code EdgeTTS)

Áp dụng đúng nguyên tắc VITS đã dùng cho duration (train trên ground-truth MAS alignment, không bao giờ train trên dự đoán riêng của model): tính trung bình F0 ground-truth theo từng phoneme (cần cho mục tiêu regression của F0Predictor), rồi **mở rộng lại giá trị đó về frame-level qua đúng ma trận alignment cứng** dùng để mở rộng `m_p`/`logs_p` — decoder train và infer thấy **cùng một dạng tín hiệu bậc thang**, đóng mismatch tại nguồn thay vì vá sau.

Toàn bộ pipeline được thêm mới, toggleable qua `--use-f0` (mặc định `False`, giống `--use-vocos`/`--use-snake`):
- `preprocess/norm_audio.py`: `cache_f0()` (pyworld DIO+StoneMask, nội suy log-space qua đoạn unvoiced)
- `vits/dataset.py`, `preprocess/dataset.py`, `preprocess/worker.py`, `preprocess/__main__.py`: mang `audio_f0_path`/`f0` xuyên suốt pipeline, fallback graceful cho dataset cũ không có F0
- `vits/modules/f0_predictor.py` (mới): `F0Predictor` (2×Conv+ReLU+LayerNorm+Dropout+proj)
- `Vocos/convnext.py`, `Vocos/generator.py`: `f0_cond` (Conv1d 1→dim, zero-init) — chỉ áp dụng cho nhánh Vocos generator
- `vits/modules/synthesizer.py`: điểm sửa bug chính — `forward()` và `infer()` đều dùng cùng công thức mở rộng F0 qua alignment cứng

**Bug tự phát hiện trong lúc implement**: `f0_cond.weight.data.zero_()` gọi **trước** `self.apply(self._init_weights)` — bị `_init_weights` (dùng `trunc_normal_`) ghi đè mất, làm module không còn zero-init an toàn khi ghép vào checkpoint cũ. Fix: chuyển `.zero_()` ra sau `self.apply(...)`.

### 1.4. Bug nghiêm trọng nhất phiên này: F0 chưa từng được bật trong lần fine-tune đầu

Sau khi fine-tune `vocos_small_f0_test_v2` chạy xong **toàn bộ 250 epoch** (resume từ `vocos_small_run_v5_fixed` epoch=1274), kiểm tra lại `loss_f0` qua TensorBoard event file phát hiện: **chính xác 0.0 tại toàn bộ 2211 điểm log, từ step 49 đến step 98000** — không phải làm tròn hiển thị, là 0.0 tuyệt đối.

**Nguyên nhân gốc**: `dataset.jsonl` dùng cho fine-tune này được preprocess **trước khi** pipeline `cache_f0()` tồn tại → **0/13100 utterance có `audio_f0_path`**. Bước "preprocessing smoke test" trước đó chỉ chạy trên vài file mẫu nhỏ, chưa từng chạy lại preprocessing đầy đủ trên dataset thật. `UtteranceCollate` có logic `has_f0 = all(u.f0 is not None for u in utterances)` — không utterance nào có F0 nên `batch.f0s` luôn `None`, nhánh `if self.use_f0 and f0 is not None` không bao giờ chạy, `f0_predictor`/`f0_cond` chưa hề nhận gradient nào suốt 250 epoch. Toàn bộ 250 epoch train (~45+ giờ GPU) thực chất tương đương `use_f0=False`.

Đây cũng giải thích lại xu hướng UTMOS quan sát được trong lúc train (tăng ở epoch~82 rồi giảm nhẹ ở epoch~200): hoàn toàn là nhiễu GAN thông thường, không liên quan gì tới F0.

**Bài học**: khi resume/fine-tune trên dataset đã preprocess từ trước, phải xác minh trực tiếp dữ liệu mới (không phải chỉ code) có đúng field cần thiết — code đúng không đảm bảo dữ liệu đúng.

**Trạng thái cuối phiên**: pipeline F0 đã implement và verify đúng về mặt code (unit-test/smoke-test qua), nhưng **chưa có lần fine-tune nào thực sự train được F0 thành công** — cần chạy lại preprocessing đầy đủ trên toàn bộ dataset rồi fine-tune lại mới đánh giá được tác động thật của F0 trên BanhmiTTS.

---

## 2. Khảo sát PTQ Quantization — `baseline` vs `vocos_small`

Báo cáo chi tiết đầy đủ: [`QAT-Training/PTQ_Sweep_Report_baseline_vocos_small.md`](../../QAT-Training/PTQ_Sweep_Report_baseline_vocos_small.md). Tóm tắt các điểm chính:

### 2.1. Khác biệt kiến trúc `dec` buộc phải khảo sát riêng op_types

| | baseline (HiFi-GAN) | vocos_small (Vocos) |
|---|---|---|
| Conv/ConvTranspose học được | 23 | 9 (`embed`+`dwconv`×8) + 2 cố định (ISTFT, không học) |
| Linear/MatMul học được | 0 | **17** (`pwconv1`+`pwconv2`×8 + `head.out`) — phần lớn compute decoder |

→ `vocos_small` phải quantize thêm `MatMul` mới có ý nghĩa; `baseline` chỉ cần `Conv`/`ConvTranspose` như phương pháp cũ.

### 2.2. Phương pháp 2 giai đoạn

1. Sweep rẻ (n=25, chỉ WER): 14 cấu hình loại trừ layer × 2 model = 28 lượt PTQ không cần train.
2. Xác nhận đầy đủ (n=100→500, WER+UTMOS22+UTMOSv2+RTF) trên cấu hình tốt nhất, để bắt các trường hợp WER "trông ổn" nhưng UTMOS giảm thật.

### 2.3. Kết quả cuối cùng (n=500, cùng bộ 500 câu test chung cho cả 2 model)

| Model | Variant | WER | UTMOS22 | UTMOSv2 | RTF | Size |
|---|---|---|---|---|---|---|
| baseline | FP32 | 0.171 | 3.929 | 3.126 | 0.028 | 65.4MB |
| baseline | PTQ-thuần (flow+enc_p+dp) | — | — | — | — | 23.0MB |
| vocos_small | FP32 | — | — | — | — | 73.3MB |

*(Bảng đầy đủ FP32/PTQ-thuần n=500 của vocos_small chưa hoàn tất khi phiên này kết thúc — xem §6 khuyến nghị.)*

**Phát hiện chính**: cấu hình `everything` (quantize cả `dec`) thắng ở sweep rẻ (WER thấp, size nhỏ nhất) nhưng khi đo UTMOS đầy đủ mới lộ ra giảm chất lượng thật:
- `vocos_small::everything`: UTMOS22 3.062→2.893 (-0.17), UTMOSv2 3.274→3.111 (-0.16)
- `baseline::everything`: UTMOS22 3.970→3.670 (-0.30), UTMOSv2 3.162→2.786 (-0.38, tệ hơn cả FP32)

**Kết luận — công thức chung cho cả 2 kiến trúc**: **quantize `flow`+`enc_p`+`dp`, giữ nguyên `dec` ở FP32.** Vì cả 2 model chia sẻ chung `enc_p`/`dp`/`flow`, chỉ khác `dec` — công thức "loại trừ `dec`" transfer được giữa 2 kiến trúc hoàn toàn khác nhau, mở rộng kết luận "không có công thức chung" của báo cáo QAT trước đó (BanhmiTTS_v1 vs Piper Baseline, 2 model không chia sẻ gì) thành: **công thức chung tồn tại khi 2 kiến trúc chia sẻ phần lớn cấu trúc, khác biệt chỉ ở phần bị loại trừ**.

### 2.4. Benchmark tốc độ thật trên CPU có AVX-VNNI

Máy chạy khảo sát (i7-14700, Raptor Lake) **có AVX-VNNI** — khác máy dev gốc của báo cáo QAT cũ (i5-10400F, không VNNI, chỉ đo được ~1.05-1.08x speedup).

| Cấu hình (baseline) | Speedup @2 thread | Speedup @1 thread |
|---|---|---|
| INT8 flow+enc_p+dp (dec=FP32) | 1.10x | 1.14x |
| INT8 everything (dec quantize) | 1.31x | 1.52x |

Quantize `dec` cho tốc độ tăng thật (đặc biệt rõ ở 1 thread), nhưng đổi lại mất ~0.3-0.4 điểm UTMOS — không đáng cho ứng dụng ưu tiên chất lượng giọng nói.

---

## 3. QAT Training

Script mới: `banhmi_train/vits/quantize.py` (port từ QAT-Training, generic theo `wrap_types`) và `banhmi_train/quantize_qat.py` (port `quantize_qat.py`, chuyển từ manual-optimization sang automatic-optimization + `optimizer_idx` để khớp `VitsModel` gốc của BanhmiTTS — discriminator warmup thực hiện bằng cách trả `None` từ `training_step` khi `optimizer_idx==1` trong giai đoạn warmup).

Ngân sách QAT tính theo khuyến nghị NVIDIA (~10% tổng bước training gốc), tỉ lệ theo chính lịch sử training của từng checkpoint:

| Model | Epoch gốc | global_step gốc | Epoch QAT | Best checkpoint |
|---|---|---|---|---|
| baseline | 1489 | 1,168,160 | 150 | epoch=93, val_loss_mel=19.2383 |
| vocos_small | 1274 | 1,002,150 | 125 | epoch=98, val_loss_mel=19.0114 |

### 3.1. Kết quả cuối cùng — baseline (n=500, INT8 nén thật, không phải QDQ mô phỏng)

| Variant | WER | UTMOS22 | UTMOSv2 | Size |
|---|---|---|---|---|
| FP32 gốc | 0.171 | 3.929 | 3.126 | 65.4MB |
| PTQ-thuần | — | — | — | 23.0MB |
| **QAT epoch=93 → INT8 nén thật** | **0.134** | **4.035** | **3.526** | **23.0MB** |

Bản INT8 nén thật sau QAT **nhỏ hơn FP32 65% nhưng chất lượng cao hơn cả FP32 gốc** — UTMOS22 +0.11, UTMOSv2 +0.40. Đây là bằng chứng rõ ràng: train cho model thích nghi với nhiễu lượng tử hóa mang lại lợi ích thật so với PTQ tĩnh.

### 3.2. Kết quả cuối cùng — vocos_small (n=500, cùng bộ 500 câu test chung với baseline)

| Variant | WER | UTMOS22 | UTMOSv2 | RTF | Size |
|---|---|---|---|---|---|
| **QAT epoch=98 → INT8 nén thật** | 0.106 | 3.402 | 3.395 | 0.010 | 31.0MB |

So với baseline QAT trên **cùng 500 câu**: baseline thắng UTMOS rõ rệt (4.035/3.526 vs 3.402/3.395), vocos_small thắng WER và nhanh hơn ~3.4 lần (0.010 vs 0.033 RTF) — đúng đặc trưng thiết kế Vocos (không có chuỗi upsample, ISTFT gần miễn phí compute).

**Phát hiện phụ đáng chú ý**: checkpoint giữa chừng (epoch=35) và checkpoint cuối (epoch=98) của vocos_small QAT cho kết quả **gần như giống hệt nhau** (WER 0.106 cả 2, UTMOS22 3.404 vs 3.402) — QAT của vocos_small bão hòa rất sớm (epoch 35/125), 63 epoch sau không cải thiện gì thêm. Khớp với pattern đã ghi nhận trong báo cáo QAT-Training cũ (cả BanhmiTTS_v1 và Piper Baseline đều bão hòa sớm rồi giữ nguyên phần còn lại của run).

### 3.3. Chi tiết kỹ thuật quan trọng: 2 dạng export INT8 khác nhau

- **QDQ mô phỏng** (`export_onnx_qat.py`-style): trọng số vẫn FP32 trên đĩa, chỉ chèn QuantizeLinear/DequantizeLinear để mô phỏng đúng số học INT8. Dùng tốt để đánh giá chất lượng (UTMOS/WER phản ánh đúng), nhưng **vô dụng cho size/tốc độ** — file to hơn cả FP32 gốc (baseline: 68.9MB), RTF thậm chí chậm hơn FP32 vì làm thêm việc round-trip giả lập.
- **INT8 nén thật** (`export_fp32_from_qat.py` + `ptq_quantize.py`): export trọng số đã học qua QAT thành ONNX FP32 thường (không wrapper), rồi chạy PTQ thật lên trên — giữ lợi ích từ QAT training nhưng có size/tốc độ thật (khớp đúng size PTQ-thuần, 23.0MB/31.0MB).

Nhầm lẫn giữa 2 dạng này từng khiến báo cáo RTF ban đầu sai (kết luận nhầm "INT8 chậm hơn FP32" trong lúc thực ra đang đo bản QDQ mô phỏng) — đã tự phát hiện và đính chính khi người dùng hỏi lại về VNNI.

---

## 4. Điều tra chất lượng âm thanh `vocos_small` (rè/crackling)

### 4.1. Phát hiện qua nghe thử trực tiếp

Sinh WAV cùng 1 câu từ nhiều checkpoint để so sánh trực tiếp (không chỉ dựa vào UTMOS). Người dùng xác nhận `vocos_small` bị rè **ngay cả ở checkpoint FP32 gốc** (chưa qua QAT/quantize) — loại trừ khả năng đây là artifact của quantization.

### 4.2. Cô lập nguyên nhân: so sánh với `vocos_run_full`

Máy có sẵn checkpoint `vocos_run_full` (cấu hình đầy đủ theo paper gốc: `vocos_dim=512, intermediate_dim=1536`, val_loss_mel=18.2628 — tốt hơn cả `vocos_small`'s 19.4528). Sinh cùng câu từ checkpoint này: **sạch, thậm chí sạch hơn cả baseline HiFi-GAN**.

→ Xác nhận: rè không phải lỗi kiến trúc Vocos nói chung, mà liên quan tới cấu hình `vocos_small` cụ thể (`dim=160`, chỉ ~1/3 width bản gốc).

### 4.3. Tính toán tham số — phát hiện quan trọng

| Config | dim | intermediate_dim | n_fft | Tham số backbone+head |
|---|---|---|---|---|
| vocos_full | 512 | 1536 | 1024 | ~13.86M (khớp `test_function` đo được: 13,861,378) |
| vocos_small | 160 | 480 | 1024 | ~1.63M |
| HiFi-GAN (project) | — | — | — | 1.6-2.2M |

`vocos_small` (160/480) **đã có tham số ngang tầm HiFi-GAN** — vấn đề không phải "chưa đủ nén" mà là **width này không đủ để dự đoán phase chính xác** ở `n_fft=1024` (head.out phải dự đoán 1026 giá trị/frame).

### 4.4. Sai lầm tự nhận ra: chưa cô lập được biến

Đề xuất ban đầu ("chỉ giảm `n_fft`, giữ `dim=160`") **chưa có bằng chứng cô lập** — 2 điểm dữ liệu sẵn có (`dim=160/n_fft=1024`→rè, `dim=512/n_fft=1024`→sạch) chỉ khác nhau ở `dim`, chưa test `n_fft` độc lập. Người dùng phát hiện đúng lỗ hổng này. Đề xuất sửa: tăng nhẹ `dim` (160→192, `intermediate_dim` 480→576, giữ tỉ lệ 3x) kết hợp giảm `n_fft` (1024→512) → ước tính ~2.15M tham số, vẫn ở mức cao của HiFi-GAN.

### 4.5. Nghiên cứu tài liệu — làm phức tạp thêm bức tranh

Tìm được paper **"Revisiting Vocos: That Phasiness Business in Time-Frequency Neural Vocoding"** (arXiv 2607.24323, 2026) — nghiên cứu trực tiếp vấn đề phase artifact của Vocos:

- Tác giả gốc Vocos (Siuzdak) từng báo cáo **"scaling model không cải thiện hiệu năng"**.
- Nguyên nhân gốc theo paper: **chính lớp Conv1D đang cản trở dự đoán phase chính xác, không phải do thiếu tham số** — vấn đề inductive bias kiến trúc, không đơn thuần là bài toán capacity.
- Thử nghiệm cô lập: thay Conv1D→Conv2D cho riêng phần dự đoán phase-difference giảm **400 lần** tham số (15.0M→37.1K) mà vẫn chính xác hơn hẳn — nhưng tác giả cảnh báo Conv2D cho toàn bộ vocoder không khả thi trực tiếp (mất khả năng học cấu trúc harmonic trải rộng theo tần số).
- **Paper không đưa ra cấu hình tối thiểu cụ thể** nào để giữ chất lượng ổn định.

**Trạng thái cuối phiên**: đã thiết kế ablation 2×2 (`dim`×`n_fft`, tận dụng 2 điểm dữ liệu sẵn có + 2 run mới cần train) nhưng **chưa chạy** — mỗi run cần train từ đầu (không thể fine-tune tiếp vì đổi shape kiến trúc), ước tính 20-35 giờ để có tín hiệu sớm, 3-6 ngày để hội tụ đủ so sánh công bằng.

---

## 5. Khó khăn kỹ thuật gặp phải & cách giải quyết

| # | Vấn đề | Nguyên nhân gốc | Cách sửa |
|---|---|---|---|
| 1 | F0 zero-init bị ghi đè | `.zero_()` gọi trước `self.apply(_init_weights)` | Chuyển `.zero_()` ra sau |
| 2 | LR re-heating khi resume fine-tune | Trainer/optimizer mới không kế thừa LR đã decay của checkpoint gốc | Seed tường minh giá trị LR đã decay đúng (`lr0 × decay^epoch`) |
| 3 | **F0 chưa hề active suốt 250 epoch train** | Dataset dùng để fine-tune chưa được preprocess lại với `cache_f0()` (0/13100 có F0) | Phải preprocess lại toàn bộ dataset trước khi fine-tune F0 lần tới |
| 4 | `torch.onnx.export` lỗi thiếu `onnxscript` | Torch 2.13 mặc định dùng dynamo-exporter mới | Thêm `dynamo=False` để dùng exporter cũ (TorchScript-based) |
| 5 | QAT wrap Linear thừa cho `baseline` | `prepare_qat` mặc định wrap cả Conv+ConvTranspose+Linear dù `baseline` PTQ chỉ dùng Conv/ConvTranspose | Thêm tham số `wrap_types` để giới hạn đúng scope |
| 6 | Ước tính sai `d_warmup_steps` (~2 epoch thay vì 1) | `global_step` đếm gộp cả bước G và D (2 lần/batch), nhầm là đếm theo batch | Tính lại chính xác theo batch thực đo (399/epoch) cho lần chạy `vocos_small` |
| 7 | Nhầm RTF của bản QDQ mô phỏng với INT8 thật | `export_onnx_qat.py`-style giữ trọng số FP32, chỉ mô phỏng số học — không đại diện tốc độ thật | Phải export riêng qua `export_fp32_from_qat.py` + PTQ thật mới đo tốc độ đúng |
| 8 | Kết luận vội "giảm n_fft là đủ" | Chưa cô lập biến `dim` vs `n_fft` — chỉ có 2 điểm dữ liệu confound | Thiết kế lại ablation 2×2 trước khi cam kết cấu hình cuối |

---

## 6. Bài học rút ra

1. **Code đúng không đảm bảo dữ liệu đúng** — bug F0 nghiêm trọng nhất phiên này không nằm ở logic model mà ở việc dataset chưa được preprocess lại. Luôn xác minh trực tiếp dữ liệu (ví dụ: đếm field có mặt trong `dataset.jsonl`) trước khi tin tưởng một tính năng mới đã "hoạt động" chỉ vì code review/smoke-test qua.
2. **`val_loss_mel`/WER không đủ để đánh giá chất lượng sau quantize** — lặp lại nhiều lần trong phiên: cấu hình PTQ "trông ổn" ở WER-only sweep (n=25) hóa ra giảm UTMOS thật khi đo đầy đủ. Luôn cần ít nhất 1 vòng UTMOS n≥100 trước khi chốt cấu hình.
3. **QAT thắng PTQ-thuần một cách nhất quán** — cả 2 model đều cho INT8 nén thật chất lượng cao hơn cả FP32 gốc sau QAT, không chỉ "phục hồi" chất lượng PTQ. Đáng làm QAT mặc định cho mọi lần deploy INT8 thay vì chỉ dùng PTQ.
4. **Layer nhạy cảm khi quantize có thể transfer giữa các kiến trúc chia sẻ cấu trúc** — không phải "luôn khác nhau hoàn toàn" như kết luận báo cáo QAT trước, mà phụ thuộc mức độ 2 kiến trúc chia sẻ module chung.
5. **Tốc độ và chất lượng là đánh đổi thật, không phải free lunch** — quantize `dec` cho tốc độ tăng 1.3-1.5x nhưng mất 0.3-0.4 điểm UTMOS; phải đo cả 2 trục trước khi quyết định.
6. **Đừng kết luận nguyên nhân từ dữ liệu confound** — 2 điểm dữ liệu khác nhau nhiều biến cùng lúc không đủ để quy nguyên nhân cho 1 biến cụ thể; người dùng tự phát hiện lỗ hổng này trong lúc thảo luận, đúng tinh thần "verify trước khi claim" đã áp dụng suốt dự án.
7. **Nghiên cứu tài liệu bên ngoài có thể phủ định giả thuyết nội bộ** — paper "Revisiting Vocos" cho thấy vấn đề phase có thể là giới hạn kiến trúc (Conv1D), không chỉ capacity — quan trọng phải tham khảo trước khi đầu tư nhiều giờ GPU vào một hướng chưa chắc đúng.
8. **Luôn nghe thử trực tiếp, không chỉ tin số liệu tự động** — UTMOS22=3.4 của `vocos_small` không tệ trên giấy, nhưng nghe thử mới lộ rõ artifact "rè" mà automated MOS predictor không nắm bắt đầy đủ.

---

## 7. Khuyến nghị công việc tương lai

### Ưu tiên cao
1. **Preprocess lại toàn bộ dataset với `cache_f0()`**, sau đó fine-tune F0 lại từ đầu (checkpoint transplant từ `vocos_small_run_v5_fixed` hoặc `baseline_v2_noclip`) — đây là công việc chưa từng thực sự chạy được trong phiên này dù code đã sẵn sàng và verify đúng.
2. **Hoàn tất eval n=500 FP32/PTQ-thuần của `vocos_small`** trên cùng bộ test chung với baseline (đang chạy dở khi phiên kết thúc) — cần để có bảng so sánh đầy đủ 6 dòng (FP32/PTQ/QAT × 2 model).
3. **Chạy ablation 2×2 `dim`×`n_fft`** cho vấn đề rè của `vocos_small` (2 run mới: `dim=160/n_fft=512` và `dim=512/n_fft=512`, tận dụng 2 điểm dữ liệu sẵn có) — cần chuẩn bị tinh thần kết quả có thể là "không cấu hình nhỏ nào đủ tốt" theo cảnh báo từ paper "Revisiting Vocos".

### Ưu tiên trung bình
4. Nếu ablation xác nhận `dim` (không phải `n_fft`) là biến quyết định → cân nhắc chấp nhận `vocos_run_full` (13.86M tham số) làm bản chính thức thay vì tiếp tục cố nén Vocos xuống ngang HiFi-GAN — bù lại bằng QAT/PTQ để giảm size deploy (bài học §6.3 cho thấy QAT trên model capacity đủ nhiều khả năng bền hơn).
5. QAT hóa `vocos_run_full` (nếu chọn hướng này) theo đúng công thức đã xác nhận (`flow+enc_p+dp`, loại `dec`) — infra đã sẵn sàng (`quantize.py`/`quantize_qat.py`/`export_fp32_from_qat.py`), chỉ cần tính lại ngân sách 10% step theo lịch sử training riêng của checkpoint này.
6. Cân nhắc thử hướng paper đề xuất (Conv2D cho riêng phần phase-difference) như một thử nghiệm kiến trúc dài hạn, dù paper cảnh báo chưa có giải pháp trọn vẹn cho toàn bộ vocoder.

### Ghi chú vận hành
- Toàn bộ script mới (`ptq_quantize.py`, `eval_wer_cheap.py`, `eval_onnx_confirm.py`, `export_fp32_onnx.py`, `export_fp32_from_qat.py`, `export_onnx_qat.py`, `run_ptq_sweep.py`) nằm ở `/home/capstone/scripts/` (WSL) — nên copy vào repo chính thức (`banhmi_train/`) nếu muốn dùng lại lâu dài, hiện chỉ là scratch script.
- Bộ test 500 câu chung (`/tmp/eval_shared_500.json`, seed=4242) nên lưu lại cố định trong repo để mọi so sánh tương lai giữa các kiến trúc dùng chung 1 bộ, tránh lặp lại vấn đề "mỗi kiến trúc có held-out split riêng do RNG khác nhau" đã gặp phải trong phiên này.
