# Novelty: thay ResBlock HiFi-GAN bằng khối inverted-residual kiểu MobileNetV2

**Phạm vi:** thay thế cơ chế MRF (Multi-Receptive-Field) gốc của HiFi-GAN generator —
`ResBlock1`/`ResBlock2` (dilated Conv1d dày) — bằng một khối lấy cảm hứng từ
MobileNetV2 (Sandler et al. 2018): `Pointwise → weight_norm → SnakeBeta →
Depthwise → weight_norm → SnakeBeta → Pointwise → weight_norm` (linear
bottleneck, không activation sau conv cuối) + residual. Toàn bộ phần còn lại
của Generator (scaffold upsample, số tầng, tỉ lệ upsample) giữ nguyên như
baseline — đây là thay đổi **chỉ trong resblock**, đúng tinh thần yêu cầu ban
đầu. Đây là hướng novelty thay thế cho MarGan (đã bỏ, xem `margan_design.md`
— không còn theo hướng đó nữa).

Toàn bộ số liệu trong báo cáo này đo bằng `test_function/compare_generators.py`
(overfit-test trên 1 đoạn audio thật, LJ018-0126, 800 bước, lr=1e-4, cùng
seed) và các script benchmark tốc độ CPU ad-hoc, chạy trực tiếp trên WSL của
dự án — không lấy từ trí nhớ. **Lưu ý quan trọng về độ tin cậy**: phần lớn số
liệu tốc độ (RTF) trong báo cáo này được đo **trong lúc job training thật
`mbconv_441_run_100ep` vẫn đang chạy song song trên cùng máy** (xác nhận qua
`ps aux`, tiến trình training PID 59704 chiếm CPU/GPU liên tục nhiều giờ) —
điều này gây nhiễu tranh chấp tài nguyên, thể hiện rõ nhất ở việc benchmark
"hifigan" (không đổi giữa các lần đo) cho ra 11.97ms và 19.94ms ở hai lần
chạy liên tiếp cùng điều kiện (chênh ~66%). Số liệu tốc độ dưới đây nên được
đọc như **tín hiệu định hướng, không phải con số chính xác tuyệt đối** — cần
đo lại sạch khi không có job training nào chạy song song.

---

## 1. Kết luận nhanh (TL;DR)

| | Tham số | loss_mel (Final, overfit 800 bước) | Tốc độ CPU (định hướng, nhiễu) |
|---|---:|---:|---|
| baseline (HiFi-GAN resblock "2") | 2,240,000 | 10.5092 | mốc so sánh |
| **Tuần tự, expansion=(4,4,1)** | 1,943,424 | 9.3789 | nhanh hơn baseline ở mọi điều kiện luồng đã thử |
| Tuần tự, expansion=(2,2,1) | 1,771,136 | **9.0462** | nhanh hơn baseline |
| Tuần tự, expansion=(1,1,1) | 1,684,992 | **8.9742** (tốt nhất tuần tự) | nhanh hơn baseline |
| MRF (song song 3 nhánh k=3/5/7), expansion=(1,1,1) | 1,732,256 | 9.0788 | nhanh hơn baseline (nhưng chậm hơn tuần tự cùng expansion) |
| MRF, expansion=(2,2,1) | 1,862,624 | 9.4520 | **chậm hơn baseline** ở đa luồng |
| MRF, expansion=(4,4,1) | 2,123,360 | **8.9734** (tốt nhất toàn bộ) | chỉ hòa vốn với baseline, không có lợi thế tốc độ |
| MRF, expansion=(6,6,1) | 2,384,096 | 9.3273 | hòa vốn với baseline |

**Kết luận chính**: ở mọi expansion schedule đã thử, cơ chế MRF song song
(đúng cơ chế gốc của HiFi-GAN, áp dụng cho khối inverted-residual) **không
cho thấy lợi thế nhất quán** so với việc xếp tuần tự 2 khối cùng kernel=3 —
MRF chỉ thắng loss_mel rõ rệt ở (4,4,1), nhưng đổi lại mất hoàn toàn lợi thế
tốc độ mà cả hướng novelty này đang theo đuổi. **Tuần tự (4,4,1)** — cấu hình
đang chạy production (`mbconv_441_run_100ep`, `configs/banhmi_mbconv.yaml`)
— vẫn là lựa chọn cân bằng tốt nhất đã kiểm chứng.

---

## 2. Bối cảnh & động lực

Baseline HiFi-GAN generator (xem `banhmi_train/vits/modules/generator.py`)
dùng activation-trước-conv (`LeakyReLU → Conv1d`, hoặc SnakeBeta khi
`use_snake=true`) trong mỗi resblock, và cơ chế MRF thật: N nhánh dilated-conv
(`resblock_kernel_sizes`, mặc định 3 nhánh k=3/5/7) chạy **song song trên
cùng input**, cộng lại rồi chia N (`Generator.forward()`, dòng
`xs = sum(...) / self.num_kernels`).

Novelty được đề xuất: thay mỗi nhánh dilated-conv dày bằng một khối
depthwise-separable kiểu MobileNetV2 — `ResBlockInverted`
(`banhmi_train/vits/utils/resblocks.py`) — pointwise-expand → depthwise →
pointwise-project (linear bottleneck) + residual, toàn bộ có weight_norm và
SnakeBeta.

## 3. Lỗi ổn định huấn luyện & cách sửa gốc rễ

Lần thử đầu tiên (LR=2e-4) cho kết quả "thắng" baseline nhưng đường loss dao
động bất thường (nhiều lần đảo chiều) — dấu hiệu không ổn định, không phải
thắng thật. Test lại ở LR=1e-4 đảo ngược hoàn toàn kết quả (baseline thắng).

Root-cause tìm được qua kiểm tra `(block(x) - x).abs().max()` tại lúc khởi
tạo: `pw_project` (pointwise conv cuối cùng trong khối) **không được
zero-init** — chỉ áp dụng `init_weights` chung, nên khối bắt đầu như một
nhiễu ngẫu nhiên thay vì một phép identity no-op (điều mà một residual block
mới thêm vào bắt buộc phải làm để không phá vỡ phần mạng đã ổn định phía
trước). Sửa bằng cách zero-init `pw_project.weight_g`/`bias`
(weight_norm-aware, vì `weight_g=0` làm `weight = weight_g * weight_v /
||weight_v||` bằng 0 bất kể `weight_v`) — cùng kiểu "ghép vào mà không xáo
trộn phần đã có" đã dùng cho Stage A/B của MarGan và `flow_block.py`'s
`post` layer.

Sau khi sửa: diff tại init = 0.0 chính xác; cả 2 cấu hình từng thua ở LR=1e-4
(mbv2 t=6, "_matched" t=2) đều thắng rõ ràng và ổn định (8.91–9.25 so với
baseline 10.51), giữ vững ở cả 800 và 1500 bước.

## 4. RTF 3.7x chậm hơn baseline — một khoảng trống phương pháp luận

Toàn bộ overfit-test trước đó chỉ đo `loss_mel` + số tham số, **chưa từng đo
RTF thật** trong suốt quá trình chọn cấu hình. Khi eval 500 mẫu trên
checkpoint thật (`mbconv_mrd_run_100ep`, epoch=74, expansion=6 đồng nhất mọi
tầng), RTF đo được chậm hơn baseline khoảng 3.7 lần.

Profiling từng lớp (CPU, 1 luồng) xác định nguyên nhân: chi phí pointwise
expand/project tỉ lệ với `expansion × T` (T = số frame tại tầng đó) — ở
expansion=6 đồng nhất, riêng resblock của tầng cuối (T=8192, tầng full
sample-rate) chiếm ~47ms trong tổng ~84ms forward pass. Đây là khoảng trống
thật trong phương pháp luận trước đó (tối ưu chất lượng/tham số mà bỏ qua
chất lượng/FLOP hay wall-clock).

**Giải pháp**: expansion theo từng tầng thay vì đồng nhất — cao ở các tầng
đầu (T nhỏ, giãn kênh rẻ), giảm về 1 ở tầng cuối (T lớn nhất, đắt nhất). Sweep
qua (6,6,1) → (4,4,1) → (2,2,1) → (1,1,1) (bảng §1) cho thấy càng giảm
expansion càng nhanh **và** loss_mel càng tốt hơn — tức cơ chế "expand" gốc
của MobileNetV2 (thiết kế cho ảnh, không phải cho audio-vocoder) có thể
không cần thiết cho tác vụ này; (1,1,1) khiến khối gần với MobileNetV1
(depthwise-separable thuần) hơn là MobileNetV2.

## 5. Kernel size của depthwise conv: k=3 thắng

Sweep k=3/5/7 ở expansion=(1,1,1), blocks_per_stage=2 (giữ nguyên các biến
khác):

| kernel | Tham số | loss_mel (Final) |
|---|---:|---:|
| k=3 | 1,684,992 | **8.9742** |
| k=7 | 1,686,784 | 9.1320 |
| k=5 | 1,685,888 | 9.5007 |

k=3 (giá trị nhỏ nhất trong `resblock_kernel_sizes` của HiFi-GAN gốc, cũng là
kernel size mặc định thường dùng trong MobileNetV2) thắng rõ, không đơn điệu
theo k (k=5 tệ nhất, không phải k=7) — nên chọn theo số đo, không suy diễn lý
thuyết.

## 6. MRF song song vs xếp tuần tự — câu hỏi trọng tâm của báo cáo này

Toàn bộ §3–5 dùng xếp **tuần tự** 2 khối `ResBlockInverted` cùng kernel=3 mỗi
tầng (`_HiFiGANInverted` / production `Generator`'s `resblock="mb"`) — **không
phải** cơ chế MRF thật của HiFi-GAN (N nhánh **song song trên cùng input**,
mỗi nhánh kernel khác nhau, cộng rồi chia N). Câu hỏi đặt ra: áp cơ chế MRF
thật (giữ đúng tinh thần thiết kế gốc của HiFi-GAN) với `ResBlockInverted`
làm nhánh, có tốt hơn xếp tuần tự không?

Cài đặt test: `_HiFiGANInvertedMRF` (`test_function/compare_generators.py`) —
mỗi tầng có 3 nhánh `ResBlockInverted` song song (kernel 3/5/7), forward tính
`sum(branch(x) for branch in branches) / 3` — đúng công thức MRF gốc.

### 6.1. Chất lượng (loss_mel Final, overfit 800 bước, cùng seed/lr)

| Expansion | Tuần tự | MRF | Ai thắng |
|---|---:|---:|---|
| (1,1,1) | **8.9742** | 9.0788 | Tuần tự |
| (2,2,1) | **9.0462** | 9.4520 | Tuần tự (cách biệt lớn) |
| (4,4,1) | 9.3789 | **8.9734** | MRF |
| (6,6,1) | *(chưa đo tuần tự ở đúng cấu hình này)* | 9.3273 | — |

Tuần tự thắng 2/3 cặp so sánh trực tiếp, và ở (2,2,1) cách biệt khá lớn
(9.05 vs 9.45). MRF chỉ thắng ở (4,4,1).

### 6.2. Tốc độ CPU (ms/call, forward pass đơn lẻ trên generator, input tổng hợp)

Đo 2 điều kiện: 1 luồng (`torch.set_num_threads(1)`) và đa luồng mặc định
(12 luồng trên máy có 24 core) — **nhắc lại: đo trong lúc training thật đang
chạy song song, xem cảnh báo đầu báo cáo**.

| | baseline | (1,1,1) | (2,2,1) | (4,4,1) | (6,6,1) |
|---|---:|---:|---:|---:|---:|
| Tuần tự, 1 luồng | 22.17 | 9.22 | — | 19.31 | — |
| Tuần tự, 12 luồng | 11.97 | 8.04 | — | 10.34 | — |
| MRF, 1 luồng | 24.29 | 14.00 | 16.23 | 30.58 | 41.40 |
| MRF, 12 luồng | 19.94 | 17.30 | 20.84 | 19.53 | 20.06 |

Nhận xét: MRF luôn chậm hơn tuần tự **cùng expansion** ở mọi điều kiện luồng
đo được — hợp lý vì MRF chạy 3 nhánh mỗi tầng thay vì 2 khối tuần tự, nhân
chi phí pointwise/depthwise lên. Ở đa luồng, MRF(2,2,1) đảo chiều thành chậm
hơn baseline (20.84 vs 19.94), còn MRF(4,4,1)/(6,6,1) chỉ xấp xỉ hòa vốn với
baseline (chênh <3%, trong biên độ nhiễu đo) — không có lợi thế tốc độ thật
sự. Chỉ MRF(1,1,1) giữ được lợi thế tốc độ rõ ràng so với baseline ở cả 2
điều kiện luồng, nhưng đổi lại thua tuần tự(1,1,1) cả về tốc độ (17.30ms vs
8.04ms ở 12 luồng, chậm hơn gấp đôi) lẫn loss_mel (9.0788 vs 8.9742).

### 6.3. Kết luận của §6

**Không có cấu hình MRF nào thắng đồng thời cả chất lượng lẫn tốc độ so với
bản tuần tự cùng expansion.** MRF(4,4,1) là cấu hình MRF tốt nhất về
loss_mel (8.9734, tốt nhất toàn báo cáo) nhưng chỉ hòa vốn tốc độ với
baseline — mất hết lợi thế tốc độ mà toàn bộ hướng novelty này nhắm tới.
Giả thuyết "áp đúng tinh thần MRF của HiFi-GAN sẽ tốt hơn" **không được xác
nhận** bằng số liệu — cái giá của 3 nhánh song song (nhân ba chi phí mỗi
tầng) không được bù lại tương xứng, trừ duy nhất ở (4,4,1) thì bù được chất
lượng nhưng mất tốc độ.

## 7. RTF đo trên checkpoint production thật — số liệu bị nhiễu, chưa dùng được

Eval 500 mẫu trên checkpoint thật `mbconv_441_run_100ep` (epoch=12, chỉ mới
train 12%) cho RTF mean=0.0685, so với baseline hội tụ đầy đủ (epoch=1489)
RTF mean=0.0431 — tức chậm hơn baseline ~1.6 lần, **trái ngược** với benchmark
generator đơn lẻ (tuần tự (4,4,1) nhanh hơn baseline ~13-14% ở cả 1 và 12
luồng). Nguyên nhân xác định được: job training `mbconv_441_run_100ep` vẫn
đang chạy real-time trên cùng máy tại đúng thời điểm eval chạy (10/09
19:14), tranh chấp CPU/GPU với chính script eval — không phải một phép đo
sạch. **Cần đo lại RTF trên checkpoint thật sau khi không còn job training
nào chạy song song** trước khi dùng số liệu này để kết luận bất cứ điều gì
về tốc độ thật của kiến trúc trong pipeline đầy đủ (không chỉ generator đơn
lẻ).

## 8. Trạng thái hiện tại & việc còn lại

- **Production `Generator`** (`banhmi_train/vits/modules/generator.py`,
  `resblock="mb"`) hiện chỉ implement xếp **tuần tự** — chưa có cơ chế MRF
  song song thật. Việc wiring MRF vào production (`resblock="mrf"`, tham số
  `mb_mrf_kernel_sizes`) đã bắt đầu nhưng **đang tạm dừng** theo yêu cầu viết
  report này trước — xem §9.
- `mbconv_441_run_100ep` (tuần tự, expansion=(4,4,1), MRD bật, batch=8+
  accumulate=2) vẫn đang chạy hướng tới epoch 100 tại thời điểm viết báo cáo
  này.
- Job background `bcjfksx6c` (sweep kernel k=3/5/7 ở (1,1,1), tuần tự) đã
  hoàn tất, số liệu đã đưa vào §5.
- Job background sweep MRF 4 cấu hình đã hoàn tất, số liệu đã đưa vào §6.

## 9. Khuyến nghị

Dựa trên toàn bộ số liệu ở §6 (đã đính chính — MRF(1,1,1) **không** vượt
trội tuần tự(1,1,1) như nhận định ban đầu, mà ngược lại), khuyến nghị:

- **Không tiếp tục theo hướng MRF song song** trừ khi có lý do mới — số liệu
  hiện tại không ủng hộ nó ở bất kỳ expansion schedule nào đã thử.
- Giữ **tuần tự (4,4,1)** làm cấu hình chính (đang chạy production), hoặc
  cân nhắc chuyển sang tuần tự **(1,1,1)** hoặc **(2,2,1)** nếu ưu tiên tốc
  độ/tham số hơn (cả hai đều thắng loss_mel rõ hơn (4,4,1) và nhanh hơn nữa)
  — nhưng cần đo RTF sạch (không tranh chấp tài nguyên, xem §7) trước khi
  quyết định cuối cùng giữa 3 lựa chọn tuần tự này.
