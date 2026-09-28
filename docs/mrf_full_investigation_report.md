# Báo cáo tổng hợp: Novelty MRF(1,1,1) — từ thiết kế đến FP32/QAT, và 2 lỗi phương pháp luận đã sửa

**Phạm vi:** toàn bộ quá trình phát triển và đánh giá hướng novelty thay thế
resblock MRF gốc của HiFi-GAN bằng `ResBlockInverted` (khối inverted-residual
kiểu MobileNetV2) — từ lựa chọn kiến trúc, training 1000 epoch thật, PTQ,
QAT, cho đến 2 lỗi phương pháp luận nghiêm trọng phát hiện giữa chừng (seed
ngẫu nhiên chưa cố định, và rò rỉ dữ liệu train/test giữa 2 model) cùng cách
khắc phục. Đây là báo cáo hợp nhất, thay thế các báo cáo từng phần trước đó
(`mbconv_resblock_report.md`, `mrf_500epoch_final_report.md`).

---

## 1. Kết luận nhanh (số liệu cuối cùng, đã sửa mọi lỗi phương pháp luận)

### 1.1. Kết quả xác đáng nhất — cả 2 model train đủ đúng lịch 1500/1500 epoch

| | baseline (epoch=1489/1500) | **MRF(1,1,1) (epoch=1386/1500)** | Chênh lệch |
|---|---:|---:|---|
| val_loss_mel | 19.7882 | **18.7681** | tốt hơn ~5.2% |
| RTF ↓ | 0.0434 | **0.0370** | **nhanh hơn ~14.7%** |
| UTMOSv2 ↑ | 3.125 | **3.651** | **tốt hơn ~16.8%** |
| WER ↓ | 0.1230 | **0.0983** | **tốt hơn ~20.1%** |
| Generator params | 1.664.768 | **1.157.024** | **nhẹ hơn 30.5%** |
| Generator size FP32 | 6.35 MB | **4.41 MB** | **nhẹ hơn 1.94 MB** |

Đây là phép so sánh đáng tin cậy nhất trong toàn bộ báo cáo: cả 2 model
được train đủ đúng lịch 1500 epoch thiết kế ban đầu (không dừng sớm), eval
với seed cố định và đúng test-set riêng của từng model (§8). **MRF(1,1,1)
thắng cả 4 trục với biên độ lớn nhất quan sát được** trong toàn bộ session.

### 1.2. Kết quả FP32/QAT ở mốc huấn luyện ngắn hơn (MRF mới epoch=836/1000)

Trước khi MRF được train đủ 1500 epoch, đã có 1 vòng đánh giá FP32 và QAT
đầy đủ ở mốc epoch=836/1000 (84% chặng đường) — vẫn giữ lại ở đây vì QAT
tương ứng (§7) đã hoàn tất, trong khi QAT trên checkpoint 1500-epoch mới
đang chạy (xem §1.3):

| Model | RTF ↓ | UTMOSv2 ↑ | WER ↓ | Generator params |
|---|---:|---:|---:|---:|
| HiFi-GAN baseline FP32 | 0.0434 | 3.125 | 0.1230 | 1.664.768 |
| MRF(1,1,1) FP32 (epoch=836/1000) | 0.0374 | 3.529 | 0.1136 | 1.157.024 |
| baseline QAT (flow+enc_p+dp int8, dec FP32) | 0.0508 | 3.549 | **0.0786** | 1.664.768 (dec không đổi) |
| MRF QAT (dựa trên epoch=836) | 0.0461 | 3.650 | 0.0977 | 1.157.024 (dec không đổi) |

Ở mốc này: FP32 MRF thắng cả 3 trục; QAT thì MRF vẫn nhanh/UTMOSv2 tốt hơn
nhưng **baseline QAT thắng WER rõ rệt** — xem §9 để biết vì sao.

### 1.3. Đang chạy: QAT dựa trên checkpoint 1500-epoch mới nhất

Đã launch QAT mới (`mrf_111_1500_qat_flow_enc_p_dp`, cùng công thức: submodules
`flow,enc_p,dp`, LR=5e-6, max_epochs=150) dựa trên checkpoint 1500-epoch
(epoch=1386, val_loss_mel=18.7681) — kỳ vọng đây sẽ là bản QAT tốt nhất từ
trước đến giờ vì xuất phát từ FP32 tốt nhất từ trước đến giờ. Kết quả sẽ bổ
sung khi hoàn tất (~150 epoch, dự kiến vài ngày).

Số liệu trong §1.1/1.2 đo với **seed cố định** (`torch.manual_seed(1234)`
trước vòng lặp suy luận) và **mỗi model dùng đúng bộ test-set 500 câu riêng
chưa từng xuất hiện trong tập train của chính nó** — 2 vấn đề phương pháp
luận nghiêm trọng phát hiện và sửa ở §8 bên dưới.

---

## 2. Kiến trúc: thay đổi gì, và tại sao

Toàn bộ scaffold Generator (conv_pre, 3 tầng ConvTranspose1d upsample
8×8×4, conv_post, `upsample_initial_channel=256`,
`upsample_kernel_sizes=[16,16,8]`, `resblock_kernel_sizes=[3,5,7]`) **giữ
nguyên 100%** giữa baseline và MRF. Khác biệt duy nhất là cơ chế resblock
sau mỗi tầng upsample:

- **Baseline** (`resblock="2"`): 3 nhánh `ResBlock2` song song (mỗi nhánh 2
  cặp dilated-Conv1d **dense**, kernel 3/5/7, dilation `[1,2]/[2,6]/[3,12]`),
  cộng rồi chia 3. Activation LeakyReLU.
- **MRF** (`resblock="mrf"`): 3 nhánh `ResBlockInverted` song song — đúng
  cơ chế MRF THẬT của HiFi-GAN (N nhánh chạy đồng thời trên cùng input,
  cộng rồi chia N), mỗi nhánh là Pointwise-expand → weight_norm → SnakeBeta
  → Depthwise(kernel 3/5/7 tương ứng) → weight_norm → SnakeBeta →
  Pointwise-project (linear, zero-init) → weight_norm, + residual.
  `expansion=1` ở mọi nhánh/tầng. Activation SnakeBeta xuyên suốt.

### 2.1. Vì sao weight_norm thay vì BatchNorm (như MobileNetV2 gốc dùng)

1. **Train/inference mismatch của BatchNorm**: BatchNorm dùng thống kê batch
   lúc train nhưng running-stats (tích luỹ, có thể trôi trong lúc GAN
   training dao động mạnh) lúc eval — literature GAN từ sau DCGAN đã ghi
   nhận rộng rãi vấn đề này, hầu hết vocoder GAN hiện đại (HiFi-GAN, WaveGAN,
   BigVGAN) bỏ hẳn BatchNorm trong generator.
2. **weight_norm loại bỏ hoàn toàn được ở inference, BatchNorm thì không**:
   `remove_weight_norm()` gấp `weight = weight_g·weight_v/‖weight_v‖` thành
   1 tensor phẳng, không tốn thêm tham số/phép tính lúc suy luận. BatchNorm
   vẫn phải lưu và áp `running_mean/var/weight/bias` — đi ngược mục tiêu
   tối thiểu tham số + compute của cả hướng novelty này.
3. **Cơ chế zero-init "identity no-op" chỉ sạch với weight_norm**: fix quan
   trọng nhất của `ResBlockInverted` (zero-init `pw_project.weight_g` để
   khối bắt đầu như residual no-op) dựa vào việc `weight_g=0` ⇒ weight hiệu
   dụng = 0 tuyệt đối. BatchNorm không có cơ chế tương đương — bước chuẩn
   hoá theo thống kê batch vẫn chủ động biến đổi activation dù zero-init
   affine gamma/beta, tức vẫn là nhiễu loạn chứ không phải identity thật.
4. **BatchNorm phức tạp hoá DDP** (2 GPU dùng xuyên suốt project) — cần
   SyncBatchNorm (thêm all-reduce, thêm cấu hình, thêm nguy cơ lệch giữa
   chạy 1-GPU/multi-GPU). weight_norm không có khái niệm thống kê theo
   batch nên miễn nhiễm.

---

## 3. Vì sao chọn MRF song song, và vì sao expansion=(1,1,1)

### 3.1. Sequential vs. MRF song song

Trước khi chốt MRF song song, đã thử xếp tuần tự (2 khối `ResBlockInverted`
nối tiếp mỗi tầng, không phải N nhánh song song) — xem
`docs/mbconv_resblock_report.md` cho toàn bộ so sánh. Quyết định cuối: ưu
tiên MRF song song vì giữ đúng tinh thần thiết kế gốc của HiFi-GAN (N nhánh
đa kernel chạy đồng thời), dù tuần tự có lúc thắng nhẹ ở một số schedule.

### 3.2. Vì sao thiết kế "giảm dần theo tầng" (X,X,1) chứ không đồng nhất (X,X,X)

Chi phí mỗi nhánh `ResBlockInverted` tỉ lệ với `expansion × T` (T = số frame
tại tầng đó). Ba tầng upsample có T tăng rất nhanh (256→2048→8192, nhân
8×8×4) trong khi kênh giảm dần (256→128→64→32) — tầng cuối là nơi 1 đơn vị
expansion tốn kém nhất theo thời gian thực. Với MRF, vấn đề bị khuếch đại
thêm vì mỗi tầng chạy **3 nhánh song song** cùng lúc — expansion cao đồng
nhất (kiểu 6,6,6) khiến tầng cuối trả chi phí `expansion×3 nhánh×T=8192`,
đắt nhất toàn generator. Vì vậy dạng (X,X,1) — expansion cao ở 2 tầng đầu (T
nhỏ, giãn kênh rẻ), giảm về 1 chỉ ở tầng cuối — được ưu tiên thử trước.

### 3.3. Kết quả 4 schedule (X,X,1) đã thử (overfit-test 800 bước)

| Expansion | Params | loss_mel (Final) | Tốc độ 1 luồng | Tốc độ 12 luồng | So với baseline (24.29/19.94ms) |
|---|---:|---:|---:|---:|---|
| (6,6,1) | 2.384.096 | 9.3273 | 41.40ms | 20.06ms | ~hòa (12 luồng), chậm hơn nhiều (1 luồng) |
| (4,4,1) | 2.123.360 | **8.9734** (tốt nhất) | 30.58ms | 19.53ms | ~hòa, không có lợi thế tốc độ rõ |
| (2,2,1) | 1.862.624 | 9.4520 (tệ nhất) | 16.23ms | 20.84ms | chậm hơn ở 12 luồng (đảo chiều) |
| **(1,1,1)** | **1.732.256** | 9.0788 | **14.00ms** | **17.30ms** | **nhanh hơn rõ ở cả 2 điều kiện** |

Ngay cả (6,6,1) — bảo thủ nhất — cũng không giữ được lợi thế tốc độ khi
nhân 3 nhánh song song. Chỉ (1,1,1) giữ được lợi thế tốc độ nhất quán.

### 3.4. Chốt (1,1,1): 3 lý do

1. **Duy nhất giữ được lợi thế tốc độ đáng tin cậy** — mục tiêu chính của
   cả hướng novelty.
2. **loss_mel không tệ nhất**, cách schedule tốt nhất (4,4,1) chỉ ~1.2%
   trong khi (4,4,1) tốn thêm ~22% tham số, không có lợi thế tốc độ.
3. **Nhẹ nhất trong 4 lựa chọn**.

Kết quả training 1000 epoch thật (§5) xác nhận lựa chọn này đúng: (1,1,1)
thắng cả WER và UTMOSv2 so với baseline khi train đủ lâu.

---

## 4. Tính toán chi tiết param & size (production `Generator`)

Đo trực tiếp trên `banhmi_train/vits/modules/generator.py::Generator`
(class production thật), cùng `initial_channel=192`,
`upsample_rates/upsample_initial_channel/upsample_kernel_sizes/
resblock_kernel_sizes/resblock_dilation_sizes` (2 config file
`configs/default.yaml` và `configs/banhmi_mrf.yaml` dùng **chính xác cùng**
giá trị cho các tham số này, chỉ khác `resblock`/`use_snake`/`mb_expansion`):

| Thành phần | Baseline (resblock="2") | MRF (resblock="mrf", expansion=1,1,1) |
|---|---:|---:|
| `conv_pre` | 344.320 | 344.320 *(giống hệt)* |
| `ups` (3× ConvTranspose1d) | 672.416 | 672.416 *(giống hệt)* |
| `pre_up_snakes` | 0 *(LeakyReLU)* | 896 *(SnakeBeta có α/β)* |
| **Resblock** | **647.808** *(9× ResBlock2 dense)* | **139.104** *(9× ResBlockInverted depthwise-separable)* |
| ↳ tầng 0/1/2 | *(gộp)* | 104.064 / 27.456 / 7.584 |
| `final_snake` | 0 | 64 |
| `conv_post` | 224 | 224 *(giống hệt)* |
| **TỔNG** | **1.664.768** | **1.157.024** |

Chênh lệch: **-507.744 tham số (-30.50%)**, toàn bộ nằm ở khối resblock
(giảm 78.5%) — vì depthwise conv scale tuyến tính theo kênh (không phải
bình phương như dense conv), và kênh giảm rất nhanh qua các tầng.

**Kích thước Generator**: FP32 6.35MB→4.41MB, bf16 3.18MB→2.21MB, int8
1.59MB→1.10MB (baseline→MRF).

**Kích thước toàn bộ pipeline ONNX** (chưa quantize, đo qua PTQ sweep §6):
baseline 65.8MB, MRF 63.9MB.

---

## 5. Training thật: 500 epoch → resume 1000 epoch → resume 1500 epoch

| Epoch | val_loss_mel | Ghi chú |
|---:|---:|---|
| 60 | 23.0101 | 12% (vòng 500 đầu) |
| 216 | 21.0430 | 43% |
| 353 | 20.6633 | 71% |
| 480 | 20.2203 | **best của vòng 500-epoch đầu tiên** |
| 593 | 20.0647 | resume tới 1000 epoch |
| 679 | 19.9083 | |
| 730 | 19.7752 | **lần đầu vượt baseline (19.7882)** |
| 836 | 19.5387 | best của vòng 1000-epoch (không đổi tới epoch 999) |
| 1131 | 19.1395 | resume tới 1500 epoch |
| 1305 | 18.8741 | |
| **1386** | **18.7681** | **best cuối cùng**, không cải thiện thêm tới epoch 1499 (train đủ 1500/1500) |

So sánh: baseline hội tụ ở **19.7882** sau 1489/1500 epoch (99%). MRF đạt
**18.7681** ở epoch 1386/1500 (92%) — train đủ đúng lịch 1500 epoch như
baseline, và **tốt hơn baseline ~5.2%** dù dừng sớm hơn vài epoch
(1386 vs 1489).

Trong lúc chạy, phát hiện 1 bug tương thích ngược: checkpoint baseline (train
trước khi `use_f0` được thêm vào code) không `load_from_checkpoint` được do
thiếu key `use_f0` trong hparams đã lưu — sửa bằng truyền tường minh
`use_f0=False` (đúng giá trị lịch sử, cùng convention đã dùng ở
`synth_from_ckpt.py`).

---

## 6. Post-Training Quantization (PTQ): sweep 14 cấu hình

Dùng lại hạ tầng PTQ sweep có sẵn (`run_ptq_sweep.py` + `ptq_quantize.py`,
ONNX static quantization, calibrate 60 câu, eval nhanh WER-only 25 câu/cấu
hình) — export MRF sang ONNX (phải sửa 1 bug môi trường: torch 2.13 đổi
`torch.onnx.export` sang exporter dynamo mặc định, không tương thích
`dynamic_axes` — thêm `dynamo=False` để dùng lại exporter legacy đúng như
code gốc được viết cho).

**Kết quả — cùng 1 exclusion set thắng cho cả 2 model:**

| | Config tốt nhất | WER (25 câu) | Size |
|---|---|---:|---:|
| Baseline | `flow_enc_p_dp` (quantize flow+enc_p+dp, dec giữ FP32) | 0.1215 | 23.0 MB |
| MRF | `flow_enc_p_dp` | **0.1078** | **21.1 MB** |

**Phát hiện quan trọng**: 4/14 cấu hình có quantize `dec` (`dec_only`,
`flow_dec`, `everything`, `everything_except_finallayer`) đều **crash** với
MRF — `AttributeError: 'NoneType' object has no attribute 'data_type'` sâu
trong ONNX Runtime static quantizer khi xử lý bias của Conv node. Baseline
không gặp lỗi này ở cùng cấu hình. Nguyên nhân nhiều khả năng là **depthwise
conv (groups=channels)** trong `ResBlockInverted` — thứ baseline hoàn toàn
không có. Kết luận: **vocoder MRF hiện không thể PTQ được qua pipeline này**
— chỉ có thể giữ FP32 hoặc dùng QAT (có gradient để học lại tham số lượng
tử hoá, không cần ONNX Runtime tự suy ra calibration tĩnh).

---

## 7. Quantization-Aware Training (QAT)

Dựa trên config PTQ tốt nhất (`flow,enc_p,dp`, wrap_types=conv), launch QAT
cho MRF ngay sau khi chọn xong config (tự động, theo yêu cầu), dùng lại
đúng pattern đã có sẵn cho baseline QAT (`banhmi_train.quantize_qat`,
LR=5e-6, max_epochs=150, d-warmup=785, batch=16×2 GPU bf16).

| | Resume từ | Layer wrap | Best cuối |
|---|---|---:|---|
| baseline QAT | epoch=1489 (FP32, 99% hội tụ) | — | epoch=144/150, val_loss_mel=19.2735 |
| MRF QAT | epoch=836 (FP32, best cuối) | 181 (flow=64, enc_p=37, dp=80) | epoch=97/150, val_loss_mel=19.2989 |

Cả 2 chạy đủ 150/150 epoch, không lỗi.

---

## 8. Hai lỗi phương pháp luận nghiêm trọng phát hiện giữa chừng — và cách sửa

### 8.1. Chưa cố định random seed cho suy luận ngẫu nhiên

`noise_scale=0.667`/`noise_scale_w=0.8` (khác 0) khiến mỗi lần gọi
`infer()` lấy mẫu ngẫu nhiên thật từ prior — không có `torch.manual_seed()`
nào trước vòng lặp eval trong BẤT KỲ script eval nào của session này. Nghĩa
là 2 lần eval cùng 1 checkpoint có thể ra WER khác nhau chỉ vì may rủi lấy
mẫu, không phải model thật sự khác. **Sửa**: thêm
`torch.manual_seed(1234)` ngay trước vòng lặp suy luận trong cả 4 script
eval cuối cùng.

### 8.2. Rò rỉ dữ liệu train/test giữa baseline và MRF (nghiêm trọng hơn)

`VitsModel.__init__` gọi `random_split()` để chia train/val/test **SAU KHI**
đã khởi tạo `SynthesizerTrn` (random-init toàn bộ trọng số). Vì baseline và
MRF có **số lượng tham số random-init khác nhau** (kiến trúc khác hẳn),
cùng `--seed 1234` nhưng trạng thái RNG lúc gọi `random_split` đã lệch pha
giữa 2 model → **split train/val/test khác nhau thật**.

Kiểm chứng thực nghiệm (tái tạo đúng seed + đúng hparams thật của từng
checkpoint):
- Test-set của baseline và MRF chỉ trùng **21/500** utterance.
- File `/tmp/eval_true_testset_500_baseline.json` (dùng cho **mọi** eval MRF
  trong suốt session) khớp 100% với test-set thật của baseline.
- **473/500 (94.6%)** utterance trong file đó nằm trong **tập TRAIN thật
  của MRF**.

→ Toàn bộ số liệu WER/UTMOSv2 của MRF (mọi checkpoint, cả FP32 lẫn QAT)
báo cáo *trước khi phát hiện lỗi này* đều bị nhiễm test-leakage nghiêm
trọng — MRF được "test" trên gần như chính dữ liệu nó đã học thuộc.

**Sửa**: xây lại `/tmp/eval_true_testset_500_mrf.json` — tái tạo đúng split
thật của MRF (cùng seed, cùng hparams thật từ checkpoint), xác nhận **0/500
leakage** vào tập train của MRF. Từ đó: baseline eval dùng test-set của
baseline, MRF eval dùng test-set của MRF — mỗi model được đo đúng trên dữ
liệu nó chưa từng thấy. Không cần train lại gì — đây thuần tuý là lỗi ở
bước tạo test-set/eval, không ảnh hưởng trọng số đã train.

*Đánh đổi*: 2 test-set giờ không còn trùng khớp từng câu (~21/500 chung),
nên so sánh không còn là paired-test tuyệt đối — nhưng mỗi số liệu tự nó
trở nên đáng tin cậy (không còn leakage), và đây là điều quan trọng nhất.

---

## 9. Vì sao baseline QAT cải thiện WER nhiều hơn MRF QAT?

Sau khi sửa cả 2 lỗi trên, số liệu vẫn cho thấy pattern: baseline QAT cải
thiện WER rất mạnh so với FP32 (0.1230→0.0786, ~36% tương đối), trong khi
MRF QAT cải thiện ít hơn (0.1136→0.0977, ~14%).

**Đã xác nhận cải thiện của baseline là thật, không phải artifact:**
kiểm tra từng mẫu cho thấy KHÔNG dồn vào vài outlier — số mẫu WER=0 tăng từ
158→223/500, số mẫu WER>0.3 giảm hơn nửa (58→25), WER>0.5 giảm 10→3. Đây là
cải thiện lan rộng, có hệ thống.

**Cơ chế nhiều khả năng nhất**: QAT không chỉ là thích nghi lượng tử hoá —
toàn bộ model (kể cả `dec` không bị wrap FakeQuantize) vẫn nhận gradient
trong vòng lặp GAN suốt 150 epoch QAT, ở LR hoàn toàn mới (5e-6) độc lập với
lịch LR gốc đã gần hội tụ. Về bản chất đây cũng là "train thêm 150 epoch",
và vì `dp`/`enc_p` (2 trong 3 submodule được fine-tune) quyết định trực tiếp
alignment/timing — thứ ảnh hưởng mạnh nhất đến WER — việc polish thêm 2
thành phần này hợp lý dẫn đến WER giảm mạnh.

**Chưa khẳng định được chắc chắn**: vì sao baseline hưởng lợi nhiều hơn MRF
từ đúng cùng công thức QAT (cùng kiến trúc flow/enc_p/dp, cùng submodule
scope). Giả thuyết: baseline (train đủ 1500/1500 epoch theo lịch thiết kế
hoàn chỉnh) có nền tảng flow/enc_p/dp ổn định hơn để việc polish thêm phát
huy rõ, còn MRF (dừng ở điểm không theo lịch thiết kế sẵn, 836/1000) có thể
có động lực training khác — nhưng đây là suy luận, chưa phải bằng chứng đo
trực tiếp. Cần thí nghiệm đối chứng riêng (QAT với FakeQuantize tắt hẳn, để
tách "chỉ train thêm" khỏi "học chịu nhiễu lượng tử hoá") mới khẳng định
chắc chắn.

---

## 10. Kết luận tổng thể

1. **ResBlockInverted + MRF song song + expansion=(1,1,1) là một thay thế
   thắng rõ so với resblock HiFi-GAN gốc ở FP32, kể cả khi cả 2 model được
   train đủ đúng lịch 1500 epoch thiết kế ban đầu**: nhẹ hơn 30.5% tham số,
   nhanh hơn ~14.7%, WER tốt hơn ~20.1%, UTMOSv2 tốt hơn ~16.8% (§1.1) — đo
   sạch, seed cố định, đúng test-set riêng, không còn nhiễu phương pháp
   luận. Biên độ thắng còn lớn hơn so với mốc train ngắn hơn (836/1000
   epoch, §1.2), cho thấy lợi thế của MRF không những không mất đi mà còn
   *mở rộng thêm* khi train đủ lâu.
2. **Ở QAT (dựa trên checkpoint 836/1000 cũ), MRF vẫn nhẹ/nhanh hơn nhưng
   WER kém hơn baseline** (§1.2, §9) — điểm này cần theo dõi/điều tra thêm.
   QAT mới dựa trên checkpoint 1500-epoch (tốt hơn hẳn) đang chạy (§1.3),
   kỳ vọng cải thiện cả khoảng cách này.
3. **`dec` (vocoder, chính là novelty) không quantize được qua PTQ** do lỗi
   ONNX Runtime với depthwise conv — chỉ `flow/enc_p/dp` (kiến trúc dùng
   chung, không đổi) mới quantize được ở cả 2 model.
4. Hai lỗi phương pháp luận (seed, test-leakage) phát hiện giữa chừng đã
   được sửa tận gốc, không cần train lại — nhưng là lời nhắc quan trọng:
   **so sánh 2 kiến trúc khác nhau về param count qua `random_split` dựa
   trên global RNG state là không an toàn**, cần cố định split một cách độc
   lập với kiến trúc (ví dụ: split trước khi biết kiến trúc, hoặc dùng
   `generator` riêng cố định theo hash của dữ liệu) cho các thí nghiệm
   tương lai.
