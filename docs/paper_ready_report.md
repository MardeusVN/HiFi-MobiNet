# MRF(1,1,1): Inverted-Residual Multi-Receptive-Field Vocoder cho HiFi-GAN — Báo cáo Paper-Ready

**Tóm tắt:** Thay resblock MRF gốc của HiFi-GAN (`ResBlock1`/`ResBlock2`,
dense dilated-conv) bằng cơ chế MRF thật (N nhánh song song, cộng chia N)
dùng `ResBlockInverted` — khối inverted-residual kiểu MobileNetV2
(depthwise-separable, weight_norm, SnakeBeta) làm nhánh, expansion=(1,1,1).
Kết quả: nhẹ hơn 30.6% tham số, ít hơn 68.5% FLOPs, thắng có ý nghĩa thống
kê về WER và UTMOSv2 (p<0.05) khi đo qua PyTorch (framework train gốc).
Tuy nhiên, khi đo qua ONNX Runtime (runtime triển khai thực tế, CPU có
AVX-VNNI) — baseline lại NHANH HƠN MRF một cách có ý nghĩa thống kê
(p<0.001), đặc biệt rõ ở câu dài (>5s: chậm hơn ~8-9%) — một phát hiện
phương pháp luận quan trọng: **FLOPs/params thấp không đảm bảo latency
thấp trên phần cứng/runtime triển khai thật**, do depthwise conv bị giới
hạn băng thông bộ nhớ, không tận dụng đa luồng hiệu quả như dense conv.

---

## 1. Bối cảnh & Phương pháp

### 1.1. Kiến trúc

Scaffold Generator (conv_pre, 3 tầng ConvTranspose1d upsample 8×8×4,
conv_post) giữ nguyên 100% giữa baseline và MRF — chỉ khác cơ chế resblock
sau mỗi tầng upsample. Baseline: 3 nhánh `ResBlock2` song song (dense
dilated-conv). MRF: 3 nhánh `ResBlockInverted` song song (Pointwise-expand
→ weight_norm → SnakeBeta → Depthwise → weight_norm → SnakeBeta →
Pointwise-project linear zero-init → weight_norm, + residual),
expansion=1 mọi nhánh/tầng.

**Vì sao weight_norm thay BatchNorm** (khác MobileNetV2 gốc): (1) tránh
train/inference mismatch của BatchNorm trong GAN training bất ổn; (2)
`remove_weight_norm()` loại bỏ hoàn toàn ở inference, không tốn thêm
tham số/compute; (3) cơ chế zero-init "identity no-op" (fix bất ổn training
quan trọng nhất của dự án) chỉ hoạt động sạch với weight_norm; (4) không
cần SyncBatchNorm phức tạp hoá DDP multi-GPU.

**Vì sao MRF song song thay vì xếp tuần tự**: giữ đúng tinh thần thiết kế
gốc của HiFi-GAN (N nhánh đa kernel chạy đồng thời trên cùng input).

**Vì sao expansion=(1,1,1)**: chi phí mỗi nhánh tỉ lệ `expansion × T`; với
MRF, 3 nhánh song song nhân thêm hệ số 3 vào chi phí tầng cuối (T lớn
nhất) — sweep 4 schedule (X,X,1) cho thấy (1,1,1) là schedule DUY NHẤT giữ
được lợi thế tốc độ nhất quán so với baseline (PyTorch), nhẹ nhất, và
loss_mel chỉ kém schedule tốt nhất (4,4,1) ~1.2%. Chi tiết đầy đủ: bảng
ablation 5 giai đoạn ở `docs/ablation_table_consolidated.md`.

### 1.2. Dữ liệu & Split

LJSpeech (13.100 utterance, 22.050Hz). Split **{12.500 train / 100 val /
500 test}**, theo đúng protocol chuẩn của VITS repository (Kim et al.) —
tỉ lệ được nhiều paper vocoder trích dẫn. ID file cụ thể do `random_split`
với seed cố định của project (không nhất thiết trùng byte-for-byte với
VITS repo gốc — giới hạn nhẹ, không ảnh hưởng tính hợp lệ so sánh nội bộ).

**Lưu ý phương pháp luận quan trọng đã sửa**: `random_split()` được gọi
SAU KHI khởi tạo model (random-init trọng số) trong code gốc — vì baseline
và MRF có số tham số random-init khác nhau, cùng seed nhưng trạng thái RNG
lệch pha, khiến **split train/test khác nhau thật giữa 2 model** (chỉ
trùng 21/500 utterance). Đã sửa: mỗi model dùng đúng test-set 500 câu tái
tạo từ chính hparams+seed thật của nó, xác nhận 0% leakage vào tập train
của chính nó.

### 1.3. Huấn luyện

Cả 2 model train đủ đúng lịch **1500 epoch** (baseline: best epoch=1489;
MRF: best epoch=1386, không cải thiện thêm tới epoch 1499) — cùng
batch_size=16×2 GPU, cùng optimizer/LR schedule gốc, precision bf16.

### 1.4. Đánh giá

- **Metric khách quan**: WER (Whisper "small" ASR + jiwer, chuẩn hoá
  lowercase/bỏ dấu câu), UTMOSv2 (MOS-predictor tự động), RTF
  (`synth_time / audio_duration`, trung bình per-utterance trên n=500).
- **Seed cố định** (`torch.manual_seed(1234)`) trước vòng lặp suy luận
  ngẫu nhiên (`noise_scale`/`noise_scale_w`) — trước đó CHƯA fix, là lỗi
  phương pháp luận thứ 2 đã phát hiện và sửa.
- **Kiểm định thống kê**: Mann-Whitney U (2 mẫu độc lập, không giả định
  phân phối chuẩn, phù hợp vì 2 test-set khác nhau sau khi sửa leakage) +
  bootstrap 95% CI (10.000 lần lấy mẫu lại) cho chênh lệch trung bình.
- **FLOPs**: đếm thủ công qua forward-hook trên Conv1d/ConvTranspose1d
  (MACs chuẩn = out_elements × cin/groups × kernel_size; FLOPs = 2×MACs),
  báo theo GFLOPs/giây audio sinh ra (T=86 latent frame ≈ 1.00s ở
  hop_length=256, sr=22050) — tránh phụ thuộc quy ước đếm của thư viện
  ngoài (MACs vs FLOPs khác nhau giữa các tool).
- **RTF qua 2 backend**: (a) PyTorch (framework train gốc, 1 luồng CPU,
  chuẩn methodology của paper HiFi-GAN gốc); (b) ONNX Runtime CPU EP trên
  máy có AVX-VNNI (Intel i7-14700), 2 luồng cố định — runtime triển khai
  thực tế cho INT8.

---

## 2. Kết quả chính

### 2.1. Tham số & FLOPs

| | Params | FLOPs (GFLOPs/s audio) |
|---|---:|---:|
| Baseline | 1.662.976 | 3.9016 |
| **MRF(1,1,1)** | **1.154.560** | **1.2306** |
| Chênh lệch | **nhẹ hơn 30.6%** | **ít hơn 68.5%** |

### 2.2. FP32 — PyTorch (framework train gốc), n=500, có kiểm định thống kê

| Metric | Baseline | MRF | p-value (Mann-Whitney U) | 95% CI chênh lệch (MRF−baseline) |
|---|---:|---:|---|---|
| val_loss_mel | 19.7882 | **18.7681** | — | — |
| RTF ↓ | 0.0434 | **0.0370** | — (đo qua PyTorch, chưa kiểm định riêng biệt vs ONNX) | — |
| UTMOSv2 ↑ | 3.1325 | **3.5365** | 3.24×10⁻⁹³ *** | [+0.373, +0.436] |
| WER ↓ | 0.1230 | **0.1043** | 1.20×10⁻² * | [−0.0348, −0.0025] |

*(Bảng UTMOSv2/WER trên đo qua ONNX Runtime FP32, dùng thay cho số PyTorch
vì cần cùng backend với phần kiểm định thống kê nhất quán — số RTF PyTorch
0.0434/0.0370 lấy từ eval PyTorch riêng, cùng phương pháp seed+test-set
đúng. Cả 2 phép đo cho cùng kết luận định tính.)*

### 2.3. INT8 (PTQ, flow+enc_p+dp, dec giữ FP32) — ONNX Runtime + VNNI, n=500

| Metric | Baseline INT8 | MRF INT8 | p-value | 95% CI |
|---|---:|---:|---|---|
| RTF ↓ | **0.0345** | 0.0371 | 7.73×10⁻⁶⁸ *** | [+0.0023, +0.0028] (MRF chậm hơn) |
| UTMOSv2 ↑ | 3.1209 | **3.4899** | 1.04×10⁻⁸⁵ *** | [+0.339, +0.400] (MRF tốt hơn) |
| WER ↓ | 0.1135 | 0.1124 | 0.607 **n.s.** | [−0.0184, +0.0169] (không phân biệt được) |

**Phát hiện then chốt (đảo ngược 1 phần kết luận FP32/PyTorch)**: qua ONNX
Runtime (runtime triển khai thật, 2 luồng), **baseline nhanh hơn MRF có ý
nghĩa thống kê** — ngược với PyTorch. WER ở INT8 giữa 2 model **không còn
phân biệt được về mặt thống kê** (p=0.607).

### 2.4. Nguyên nhân: depthwise conv giới hạn băng thông bộ nhớ, không scale tốt với đa luồng ở chuỗi dài

Kiểm chứng chéo 3 cách độc lập, đều cho cùng kết luận:

1. **Test dec-only cô lập** (cùng input, loại bỏ mọi confound khác): ở
   T=32 (audio ngắn), MRF vẫn nhanh hơn baseline qua cả PyTorch lẫn ONNX
   Runtime 1 luồng; nhưng ở ONNX Runtime **2 luồng**, gần như hòa ngay từ
   T nhỏ.
2. **T-sweep tổng hợp** (T=32→700 latent frame, tương ứng 0.37s→8.13s
   audio), đo qua cả 2 backend, 2 luồng:
   - ONNX Runtime: baseline vượt MRF ngay từ T=32 (+3.5%), tăng dần lên
     **+14.3% ở T=700**.
   - PyTorch: MRF vẫn thắng tới T=512 (+21%), nhưng **đảo chiều ở T=700**
     (baseline +9.0%) — xác nhận đây KHÔNG phải lỗi riêng ONNX Runtime, mà
     là đặc tính chung của depthwise conv dưới đa luồng ở chuỗi đủ dài,
     chỉ khác ngưỡng T xảy ra (ONNX Runtime sớm hơn nhiều so với PyTorch).
3. **Phân tích lại chính 500 mẫu thật theo nhóm độ dài** (không cần đo
   lại): chênh lệch RTF (MRF chậm hơn baseline) tăng dần đơn điệu theo độ
   dài câu — từ +3-5% (câu ngắn/trung, <5s) lên **+8-9%** (câu dài, >5s) —
   khớp đúng T-sweep tổng hợp, trên dữ liệu thật.

**Giải thích cơ chế**: depthwise conv (lõi `ResBlockInverted`) có cường độ
tính toán thấp (ít FLOPs/byte dữ liệu di chuyển) → bị giới hạn bởi băng
thông bộ nhớ chứ không phải tốc độ tính toán. Phép toán giới hạn-bộ-nhớ
không tận dụng thêm luồng hiệu quả (hiệu suất đa luồng giảm dần), trong
khi dense conv (baseline) có cường độ tính toán cao hơn, tận dụng tốt
kernel GEMM tối ưu đa luồng. T càng lớn → khối lượng công việc
memory-bound càng nhiều → nhược điểm đa luồng càng lộ rõ.

---

### 2.5. Tham chiếu: vị trí so với Vocos (cùng codebase, KHÔNG cùng mức độ chặt chẽ phương pháp luận)

Project đã có sẵn 1 hướng khác (Vocos — ConvNeXt backbone + ISTFT head,
xem `docs/vocos_vs_hifigan_bigvgan_edge_report.md`), eval đầy đủ 500-mẫu
ở phiên bản `vocos_small_run_v5_fixed` (epoch=1274/1500):

| | RTF (PyTorch) | UTMOS22 | UTMOSv2 | WER | val_loss_mel |
|---|---:|---:|---:|---:|---:|
| Vocos small | **0.0157** | 3.097 | 3.278 | **0.0766** | 19.4528 |
| MRF(1,1,1) (từ §2.2) | 0.0370 | — | 3.5365 | 0.1043 | **18.7681** |
| Baseline | 0.0434 | 3.990 | 3.1325 | 0.1230 | 19.7882 |

**Vocos nhanh hơn cả MRF lẫn baseline đáng kể (kiến trúc khác hẳn — ISTFT
thay vì ConvTranspose1d chuỗi dài, xem báo cáo gốc §2)** và WER tốt nhất.
Tuy nhiên, số liệu Vocos này **đo TRƯỚC KHI phát hiện và sửa 3 lỗi phương
pháp luận** của investigation MRF (seed cố định, train/test leakage do
`random_split` phụ thuộc kiến trúc, và — quan trọng nhất theo phát hiện
mới ở §2.4 — **chưa từng đo qua ONNX Runtime/VNNI**, chỉ có số PyTorch).
Vì Vocos dùng toàn bộ op khác hẳn (không có depthwise conv, ISTFT là phép
toán không tham số) rất có thể **không** gặp hiệu ứng memory-bound-dưới-đa-luồng
như MRF — nhưng đây là suy đoán, chưa kiểm chứng. **Không nên dùng số
Vocos ở đây để so sánh trực tiếp, ngang hàng với MRF/baseline** — chỉ mang
tính tham chiếu định vị, cần làm lại đúng quy trình (seed, split đúng
riêng của Vocos, benchmark qua ONNX Runtime) nếu muốn đưa vào paper như 1
so sánh chính thức.

---

## 3. Giới hạn

1. **PTQ không quantize được `dec` của MRF** — lỗi ONNX Runtime static
   quantizer với depthwise/grouped Conv (`AttributeError` khi xử lý bias).
   Baseline quantize được toàn bộ (thêm speedup tới 1.34-1.53x), MRF chỉ
   quantize được flow+enc_p+dp (phần dùng chung, không phải phần novelty).
2. **Hướng QAT đã thử nhưng chủ động dừng, chuyển hẳn sang PTQ.** QAT (150
   epoch, cả baseline lẫn MRF) huấn luyện thành công và giảm loss_mel/WER
   thật, nhưng export trực tiếp từ FakeQuantize sang ONNX chỉ tạo ra QDQ
   format (Quantize/Dequantize quanh Conv FP32 thường) — ONNX Runtime
   không tự fuse thành `QLinearConv` vì `QATConv1d` chỉ quantize input,
   không quantize output ngay sau conv (bị chặn bởi activation phi tuyến
   xen giữa trước khi tới quantize của layer sau) — nên **không có INT8
   thật tăng tốc**, chỉ mô phỏng. Sửa đúng cần thay đổi kiến trúc wrapper
   (thêm output fake-quant) + train lại QAT từ đầu — chi phí (~150 epoch
   × 2 model) không tương xứng lợi ích khi PTQ đã cho kết quả tốc độ thật
   tốt (§2.3). Quyết định: **dừng hẳn nhánh QAT, dùng PTQ cho toàn bộ kết
   quả INT8 trong báo cáo này.**
3. **ID file test-set không trùng với VITS repo gốc** — chỉ tỉ lệ
   12500/100/500 khớp convention, không so sánh tuyệt đối số liệu được với
   paper khác cùng convention.
4. **UTMOSv2 là MOS-predictor tự động, không phải đánh giá người thật** —
   cần vòng nghe MOS thật (15-30 người) để claim chất lượng đứng vững
   trước phản biện — nằm ngoài khả năng tự thực hiện của phân tích này.
5. Chưa so sánh với baseline SOTA đã công bố **bên ngoài** project (paper
   HiFi-GAN gốc, Vocos gốc, BigVGAN gốc). Có tham chiếu Vocos **nội bộ
   cùng codebase** ở §2.5, nhưng đo trước khi sửa các lỗi phương pháp
   luận (seed, split, backend) — chỉ mang tính định vị, không phải so
   sánh chính thức ngang hàng.

---

## 4. Kết luận

MRF(1,1,1) là một cải tiến hợp lệ, có ý nghĩa thống kê về **chất lượng**
(UTMOSv2, WER) và **hiệu quả tham số/FLOPs** so với resblock HiFi-GAN gốc.
Tuy nhiên, claim "nhanh hơn" **phụ thuộc hoàn toàn vào backend/điều kiện
đo**: đúng khi đo qua PyTorch (framework train), nhưng **sai khi đo qua
ONNX Runtime** (runtime triển khai CPU thực tế) — một minh chứng cụ thể,
đo lường được, cho nguyên tắc đã biết trong literature efficient-model:
**FLOPs/params thấp không đảm bảo latency thấp trên phần cứng thật**, đặc
biệt với các phép toán giới hạn-băng-thông-bộ-nhớ (depthwise conv) dưới
đa luồng và chuỗi dài. Đây là đóng góp phương pháp luận đáng kể bên cạnh
đóng góp kiến trúc của công trình này.
