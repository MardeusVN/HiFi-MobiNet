# Sập NaN ở StochasticDurationPredictor: điều tra và cách sửa triệt để

**Tóm tắt:** trong suốt quá trình train SEQ(1,1,1), model liên tục bị sập
vĩnh viễn ở `dp` (StochasticDurationPredictor) — 4 lần, ở các mốc epoch
khác nhau. Bốn vòng sửa đầu đều tấn công **nguồn sinh NaN** (thêm guard cho
từng phép toán) và đều chỉ trì hoãn chứ không loại bỏ được. Vòng sửa cuối
cùng đổi hướng: chặn ở **mắt xích biến sự cố tạm thời thành hỏng vĩnh
viễn** — và điều đó khiến chế độ hỏng này không còn khả năng xảy ra về mặt
cấu trúc, bất kể NaN sinh ra từ đâu.

---

## 1. Chữ ký của lỗi

Mỗi lần sập đều có đúng một chữ ký, lặp lại không sai khác:

| Đại lượng | Biểu hiện |
|---|---|
| `loss_dur`, `loss_gen_all` | chuyển sang `nan` **đột ngột tại một step cụ thể**, rồi NaN vĩnh viễn, không bao giờ hồi phục |
| `loss_mel`, `loss_kl` | **vẫn khỏe mạnh và tiếp tục cải thiện** hàng trăm epoch sau đó |
| `infer()` | sinh audio 0.012s (256 sample), cảnh báo `StochasticDurationPredictor produced degenerate duration(s)` |
| Checkpoint | `val_loss_mel` vẫn tiếp tục "tốt lên", nên ModelCheckpoint vẫn ghi đè "best" bằng các bản đã hỏng |

**Vì sao `dec` không bị ảnh hưởng còn `dp` thì chết**: `loss_gen_all` là tổng
của nhiều thành phần, nhưng autograd lan gradient theo **đường tính toán
thực tế** của từng tham số. `dec` không hề phụ thuộc đầu ra của `dp`, nên
gradient của `dec` được tính từ nhánh riêng của nó và vẫn hữu hạn. Chỉ các
tham số nằm trên nhánh của `dp` nhận gradient NaN — và chỉ chúng chết.

Chính đặc điểm này khiến lỗi **ẩn rất lâu**: mọi chỉ số được theo dõi
(`val_loss_mel`) đều nhìn vào `dec`, nên vẫn đẹp, trong khi model thực tế đã
mất hoàn toàn khả năng sinh tiếng nói tự do.

---

## 2. Chuỗi hỏng

```
NaN xuất hiện trong forward (phép toán số học nào đó dưới bf16)
        ↓
gradient của các tham số dp trở thành NaN
        ↓
AdamW.step() GHI NaN VÀO CHÍNH THAM SỐ          ← mắt xích quyết định
        ↓
mọi forward sau đó qua tham số NaN đều ra NaN → hỏng vĩnh viễn
```

Bốn vòng sửa đầu đều nhắm vào mũi tên **thứ nhất** — vốn là trò đập chuột,
vì spline flow dưới bf16 có rất nhiều edge case khác nhau.

---

## 3. Những gì đã thử và kết quả thật

| # | Cách sửa | Kết quả |
|---|---|---|
| 1 | `clamp_min(_EPS)` cho 2 phép chia trong `_rational_quadratic_spline` (`theta = …/input_bin_widths`, `delta = heights/widths`) | Sập lại ở ~epoch 45 (trước đó ~epoch 20) |
| 2 | `--gradient_clip_val 1.0` | **Làm tệ hơn** — xem §3.1 |
| 3 | Guard `inside_interval_mask.any()` (lỗi cộng đồng đã biết, coqui-ai/TTS#1959) | Sống lâu hơn ~10 lần (đến ~epoch 470-509) nhưng vẫn sập |
| 4 | Quét toàn bộ dataset tìm dữ liệu lỗi | **0 vấn đề** trên 13.100 utterance — dataset không phải nguyên nhân |

### 3.1. Gradient clipping làm tệ hơn, không tốt hơn

`clip_grad_norm_` tính **một norm toàn cục trên tất cả tham số**, rồi nhân
**cùng một hệ số scale** cho mọi gradient. Chỉ cần một tham số của `dp` có
gradient NaN thì norm toàn cục thành NaN, và hệ số scale NaN đó lan sang
**toàn bộ** tham số — kể cả `dec`/`enc_p` vốn hoàn toàn độc lập và trước đó
vẫn an toàn.

Đo từ TensorBoard xác nhận đúng như vậy:

| Lần chạy | gradient clip | `loss_dur` | `loss_mel` |
|---|---|---|---|
| v1, v2 | không | NaN vĩnh viễn | **vẫn khỏe hàng chục nghìn step** |
| v3 | `1.0` | NaN vĩnh viễn | **NaN cùng lúc, cũng vĩnh viễn** |

→ Đã loại bỏ hoàn toàn gradient clipping khỏi hướng sửa.

### 3.2. Baseline chưa từng thực sự được sửa

Soi lại toàn bộ lịch sử TensorBoard của baseline cho thấy nó cũng sập đúng
kiểu này: `version_0` sập ở step≈38949, `version_1` (đã bật clip) sập ở
step≈341699. Chỉ `version_2` — resume từ một checkpoint sạch khác — mới sống
tới hết 1500 epoch. Nghĩa là baseline "thành công" nhờ **lặp lại việc resume
từ checkpoint sạch cho tới khi may mắn**, không nhờ một bản vá nào cả.

### 3.3. Dataset không phải nguyên nhân

Quét trực tiếp toàn bộ 13.100 tensor đã cache (`wav`, `spec`) và `phoneme_ids`
của canonical split: **0 vấn đề** — không NaN/Inf, không sequence rỗng,
không biên độ vượt ngưỡng, không câu nào có `n_frames < n_phonemes`. Dữ liệu
tĩnh hoàn toàn sạch; bất ổn phát sinh từ **tính toán động trong lúc train**.

---

## 4. Cách sửa triệt để — ba lớp

### Lớp A. Bỏ qua optimizer step khi gradient không hữu hạn — *fix cốt lõi*

`banhmi_train/vits/training.py::VitsModel.optimizer_step`

Đây chính là lưới an toàn mà `torch.cuda.amp.GradScaler` cung cấp **miễn
phí** cho fp16: kiểm tra inf/NaN trong gradient, nếu có thì bỏ hẳn step đó và
đi tiếp. Nhưng project train bằng **bf16** — và với bf16, PyTorch Lightning
**không bật GradScaler** (bf16 có dải mũ như fp32 nên không cần loss
scaling). Toàn bộ lưới an toàn đó **biến mất** mà không có cảnh báo nào.
Đúng một batch xấu là đủ giết vĩnh viễn model, trong khi cùng đoạn code đó
chạy fp16 thì chỉ bỏ một step rồi đi tiếp.

```python
optimizer_closure()          # forward + backward, gradient được điền
if mọi gradient đều hữu hạn:
    optimizer.step()
else:
    optimizer.zero_grad(set_to_none=True)   # vứt batch hỏng
    self._skipped_steps += 1                # đếm + log + cảnh báo
```

Vì sao đây là triệt để:

- Bắt **mọi** nguồn NaN — spline flow, MAS, discriminator, tràn số bf16 —
  không cần biết phép toán nào gây ra.
- Tham số **không bao giờ** có thể trở thành NaN → chế độ hỏng vĩnh viễn bị
  loại bỏ về mặt cấu trúc, không phải chỉ giảm xác suất.
- Chi phí ~0 khi bình thường (một phép `isfinite` mỗi tham số mỗi step).
- **An toàn dưới DDP không cần đồng bộ thêm**: gradient đã được all-reduce
  trong backward, NaN ở bất kỳ rank nào cũng lan qua phép trung bình, nên
  mọi rank thấy cùng gradient và độc lập đi tới cùng một quyết định.
- **Không che giấu triệu chứng**: đếm và log số step bị bỏ ra TensorBoard
  (`skipped_steps`); nếu tỉ lệ vượt 1% thì ghi `ERROR` — phân biệt rõ "vài
  batch xui" với "model đang thực sự phân kỳ".

### Lớp B. Chạy `dp` ở fp32 — chặn tại nguồn

`banhmi_train/vits/modules/duration_predictor.py::StochasticDurationPredictor.forward`

Normalizing flow nhạy cảm về độ chính xác số học theo cách phần còn lại của
model thì không: bin width của spline đến từ một phép *cumsum rồi trừ*, và
các số hạng log-determinant chia cho những đại lượng chỉ được đảm bảo khác 0
**về mặt giải tích**. Dưới mantissa ~7 bit của bf16, chúng có thể triệt tiêu
về đúng 0.

`dp` rất nhỏ so với `dec`, nên ép fp32 gần như không tốn gì. Đặt trực tiếp
trong `forward()` nên **cả 3 chỗ gọi** (NLL lúc train, reverse-sample cho
duration discriminator, và `infer()`) đều tự động được bảo vệ:

```python
with torch.autocast(device_type=x.device.type, enabled=False):
    return self._forward_fp32(...)   # cast input sang float32
```

### Lớp C. Checkpoint "best" phải phản ánh cả sức khỏe của `dp`

`banhmi_train/vits/training.py::validation_step` + `_log_audio_samples`

`val_loss_mel` là teacher-forced: nó chạy qua `dec`/`enc_p` nhưng **không bao
giờ** chạm tới reverse-sampling của `dp`. Nên một model đã hỏng `dp` vẫn có
`val_loss_mel` đẹp dần, và ModelCheckpoint vui vẻ thăng nó lên "best", ghi
đè dần các bản thật sự tốt **cho tới khi không còn bản nào sạch để resume**
(đã xảy ra thật: cả một lần chạy mất sạch checkpoint sạch theo đúng cách
này).

Sửa: `_log_audio_samples` nay trả về trạng thái sức khỏe (phát hiện audio
degenerate hoặc không hữu hạn), và `validation_step` báo
`val_loss_mel = +inf` cho epoch đó → checkpoint hỏng **không bao giờ** được
chọn làm "best", luôn còn ít nhất một bản dùng được trên đĩa.

### Lớp D (chẩn đoán). Ghi dấu vết batch gây lỗi

Mỗi step bị bỏ đều log kèm `(epoch, batch_idx, phoneme_lengths)` — đủ để
truy ngược chính xác batch nào gây ra, trả lời dứt điểm câu hỏi "có phải do
dữ liệu không" mà không cần phỏng đoán.

---

## 5. Kiểm chứng

Không chỉ kiểm tra "không làm vỡ luồng bình thường", mà kiểm chứng **lưới an
toàn thực sự bắt được NaN**, bằng cách tiêm NaN vào đúng code path thật
(`/tmp/test_nan_guard.py`):

```
[TEST] injecting NaN into generator loss at global_step=2
WARNING: Discarded optimizer step (optimizer_idx=0): non-finite gradients.
         Parameters left untouched, batch dropped. Skipped 1 of 2 steps (50.0000%).
         Batch fingerprint (epoch, batch_idx, phoneme_lengths)=(0, 1, [211, 233, …])

NaN injected into loss:      True
steps discarded by guard:    1
all model_g params finite:   True
PASS -- guard discarded the poisoned step, parameters survived intact
```

Một chi tiết đáng lưu ý phát hiện trong lúc viết test: **cộng** một hằng số
NaN vào loss *không* tạo gradient NaN (đạo hàm của hằng số bằng 0), nên phải
**nhân** để NaN thực sự nằm trên đường gradient. Lần chạy test đầu tiên
"thất bại" chính vì lý do này — bản thân guard hoàn toàn đúng.

---

## 6. Ảnh hưởng tới các kết quả đã có

**Baseline và MRF(1,1,1) không cần train lại.** Checkpoint cuối của chúng là
kết quả đã thực sự xảy ra: nếu quá trình train ra chúng từng chạm phải edge
case này, chúng đã sập ngay lúc đó và không thể tồn tại ở trạng thái khỏe
mạnh như hiện nay. Các lớp bảo vệ thêm vào sau chỉ có ý nghĩa **phòng ngừa
cho các lần train MỚI**, không làm thay đổi — và cũng không "sửa lại" được —
bất kỳ con số nào đã đo trên các checkpoint đã có.
