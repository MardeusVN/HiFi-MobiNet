# MarGan — Lightweight Periodic-Refined Vocos Vocoder

**Đổi tên từ bản nháp gốc "PV-Vocos / Snake-VocosGAN".** Bản này đã qua nhiều vòng rà soát, sửa các điểm lý thuyết còn cấn so với bản gốc (ghi rõ ở mỗi mục sửa, đối chiếu bằng chứng thực nghiệm/tài liệu đã có trong dự án — xem `docs/banhmi-tts-report-01.md` §4 và `arXiv 2607.24323`).

## 1. Tổng quan dự án

### Tên làm việc

**MarGan**

### Mục tiêu

Thiết kế vocoder kết hợp:

- **Vocos**: dự đoán complex STFT + `iSTFT`, hiệu quả cho phần magnitude/cấu trúc phổ thô.
- **HiFi-GAN**: multi-receptive-field waveform modeling + adversarial training, mạnh ở chi tiết miền thời gian/harmonic.
- **SnakeBeta**: periodic inductive bias cho tín hiệu giọng nói.

Ràng buộc kỹ thuật chính:

> **Tổng tham số suy luận: khoảng 1–2M.**

### Cơ sở thực nghiệm cho hướng đi này (không có trong bản gốc)

Ablation cô lập biến `dim`×`n_fft` trên chính `VocosGenerator` của BanhmiTTS (`test_function/compare_generators.py`, overfit 800 bước) cho kết quả:

| dim \ n_fft | 1024 | 512 |
|---|---|---|
| 512 (full, 15.0M param) | loss_mel=2.20 | 3.63 |
| 160 (small, ~2.0M param) | loss_mel=6.00 | 7.58 |

`vocos_small` (~2.0M, đã ngang ngân sách HiFi-GAN) vẫn kém bản full 2.7 lần — không có cách phân bổ lại `dim`/`n_fft` bên trong Vocos thuần để thoát giới hạn này. Đây là lý do MarGan **tách bài toán** thay vì cố nhồi thêm capacity vào đúng chỗ đã chứng minh không hiệu quả.

---

## 2. Câu hỏi nghiên cứu chính

> **Kết hợp (a) một module refine phase nhỏ ở miền tần số ngay trước `iSTFT`, và (b) một module refine waveform nhỏ, có periodic inductive bias, ở miền thời gian sau `iSTFT` — có phục hồi được phần chất lượng mà một Vocos siêu nhẹ (dim thấp) đánh mất, trong khi vẫn giữ tổng tham số ~1–2M không?**

---

## 2b. Tiêu chí thành công

Đơn giản hóa theo đúng framing đã thống nhất — chỉ cần đạt **cả 3** so với baseline `baseline` (HiFi-GAN, đã QAT: RTF≈0.033, UTMOS22≈4.035, ~1.6-2.2M tham số) là đã thành công lớn, không cần vượt `vocos_full` (15M):

| Tiêu chí | Ngưỡng | So với |
|---|---|---|
| Tham số | thấp hơn | `baseline` HiFi-GAN (~1.6–2.2M) |
| MOS/UTMOS | cao hơn | mức hiện tại của `vocos_small` (UTMOS22≈3.40) — lý tưởng là tiệm cận `baseline`/`vocos_full` |
| RTF | thấp hơn | `baseline` HiFi-GAN (RTF≈0.033) — giữ đúng lợi thế tốc độ vốn có của Vocos |

Đạt cả 3 đồng thời = MarGan chứng minh được luận điểm cốt lõi: **có thể giữ lợi thế tốc độ của Vocos, ở ngân sách tham số ngang hoặc thấp hơn HiFi-GAN, mà không phải đánh đổi chất lượng như `vocos_small` hiện tại đang phải chịu.** Đây là mốc so sánh dùng xuyên suốt mọi Phase ở §13, không cần chờ tới khi so sánh với `vocos_full`.

---

## 3. Kiến trúc đề xuất

### 3.1. Pipeline tổng thể — **đã sửa (điểm cấn #1)**

**Vấn đề bản gốc**: refiner chỉ nhận `x0` (waveform đã qua `iSTFT`) và `h` (latent đầu vào) — không có đường truy cập tới complex STFT `Ŝ` mà Vocos vừa dự đoán. Nhưng bằng chứng thật duy nhất có được (`arXiv 2607.24323`, "Revisiting Vocos") cho thấy cách sửa phase hiệu quả nhất là làm việc **trực tiếp trên phase-difference ở miền tần số** (Conv2D nhỏ, chỉ 37.1K tham số, giảm mạnh wrapping-aware loss) — không phải sửa waveform đã collapse ở miền thời gian. Sửa `x0` sau khi thông tin phase đã "hòa" vào waveform là một bước nhảy chưa có cơ sở.

**Fix**: thêm 1 giai đoạn refine phase ở miền tần số **trước** `iSTFT`, tách biệt với refiner miền thời gian sau `iSTFT`. **Quan trọng — sửa tiếp 1 lần nữa (xem khung cảnh báo dưới)**: giai đoạn này phải làm việc trên cặp `(cos φ̂, sin φ̂)` (unit-circle representation Vocos đã dùng sẵn trong `heads.py`), **không phải góc `φ̂` tuyệt đối**.

**Làm rõ lại cho đúng (đối chiếu `heads.py` thật, dòng 139-150)**: Vocos **có** dự đoán phase tuyệt đối dạng thô (`self.out()` là `nn.Linear` thường, không bound) — đây tự nó không phải vấn đề. Vấn đề paper chỉ ra cụ thể là **ép phase qua `tanh` để bound cứng về [−π,π] trước khi dùng** — chính việc bound thêm 1 lớp không cần thiết đó mới gây lỗi ở biên (mất tính wrap tự nhiên). `heads.py` thật **không** dùng `tanh` — chỉ áp `cos()/sin()` trực tiếp lên phase thô (dòng 150: `cos_p, sin_p = torch.cos(phase), torch.sin(phase)`), nên tự động wrap-safe mà không cần bound gì cả. Stage A ở đây được thiết kế để chạy **sau** bước `cos()/sin()` này (nhận `(cos φ̂, sin φ̂)` đã tính sẵn, không đụng lại vào phase thô) — vẫn đúng hướng an toàn, chỉ là lý do trích dẫn ở bản trước hơi lẫn giữa "dự đoán phase tuyệt đối" (không sao) và "bound qua tanh" (mới là vấn đề thật):

```text
                 Latent h (VITS z, 192-dim)
                       │
                       ▼
             ┌────────────────────┐
             │ Lightweight Vocos  │
             │      Backbone      │
             └─────────┬──────────┘
                       │
                       ▼
        Magnitude M̂, (cos φ̂, sin φ̂)   ← unit-circle, KHÔNG phải góc scalar
                       │
                       ▼
        ┌──────────────────────────────┐
        │ Stage A — Phase Refiner       │   miền tần số, trước iSTFT
        │ (Conv2D nhỏ, nhận (M̂,cosφ̂,   │   ~40-80K param (mục tiêu, §6)
        │  sinφ̂), xuất δc/δs → bound   │
        │  qua tanh×0.5 → cộng → re-   │
        │  project về unit circle)      │
        └──────────────┬────────────────┘
                       │
              (cos φ̂', sin φ̂') đã refine, |·|≈1 (không đúng tuyệt đối — xem §9)
                       │
                       ▼
                Complex STFT Ŝ'
                       │
                       ▼
                     iSTFT
                       │
                       ▼
              Waveform x₀ (đã có phase tốt hơn)
                       │
                       ▼
        ┌──────────────────────────────┐
        │ Stage B — Periodic Waveform   │   miền thời gian, sau iSTFT
        │ Refiner (DWConv đa-dilation   │   ~300-500K param
        │  + SnakeBeta + 1×1 fusion)    │
        └──────────────┬────────────────┘
                       │
                       ▼
                  Residual r
                       │
                       ▼
                x = x₀ + r
                       │
                       ▼
                Waveform cuối
```

Công thức (đã sửa để wrap-safe — thao tác trên `(cos, sin)`, không phải góc):

\[
(\delta_c, \delta_s) = \varphi_A(\hat{M}, \cos\hat\varphi, \sin\hat\varphi) \qquad \text{(raw, không bound)}
\]
\[
\Delta_c = k\tanh(\delta_c) \qquad \Delta_s = k\tanh(\delta_s) \qquad k = 0.5
\]
\[
c' = \cos\hat\varphi + \Delta_c
\qquad
s' = \sin\hat\varphi + \Delta_s
\]
\[
(\cos\hat\varphi', \sin\hat\varphi') = \frac{(c',\,s')}{\sqrt{c'^2+s'^2+\epsilon}}
\qquad \epsilon = 10^{-8}
\]

**Vì sao cần bound `(Δc,Δs)` qua `tanh×k` — sửa 1 claim sai ở bản trước.** Bản trước khẳng định "output luôn nằm trên (xấp xỉ) unit circle bất kể `(Δc,Δs)`" — **sai**: nếu `(Δc,Δs) ≈ (−cos φ̂,−sin φ̂)` (correction triệt tiêu gần hết tín hiệu gốc), thì `c'²+s'²→0`, và tỉ số `(c'²+s'²)/(c'²+s'²+ε) → 0` (không phải →1) — `ε` từ "không đáng kể" thành **chi phối hẳn**, output suy biến về gốc tọa độ thay vì unit circle.

**Chứng minh bound sửa lỗi này**: với `Δc=k\tanh(δc)`, `Δs=k\tanh(δs)`, ta có `|Δc|≤k`, `|Δs|≤k` (tính chất `tanh`), nên `‖(Δc,Δs)‖_2 ≤ k\sqrt2`. Với `k=0.5`: `‖(Δc,Δs)‖_2 ≤ 0.5\sqrt2 ≈ 0.707 < 1`. Áp bất đẳng thức tam giác (chiều ngược, `‖a+b‖≥‖a‖−‖b‖`), và biết `‖(\cos\hat\varphi,\sin\hat\varphi)‖_2=1`:

\[
\sqrt{c'^2+s'^2} = \|(\cos\hat\varphi,\sin\hat\varphi)+(\Delta_c,\Delta_s)\| \geq 1 - \|(\Delta_c,\Delta_s)\| \geq 1-0.5\sqrt2 \approx 0.293
\]

\[
\Rightarrow\quad Q := c'^2+s'^2 \;\geq\; (1-0.5\sqrt2)^2 \approx 0.0858 \gg \epsilon=10^{-8}
\]

(đặt tên `Q:=c'^2+s'^2` từ đây — dùng lại nguyên vẹn ở §9 khi chặn sai số của `L_φ`, không định nghĩa lại). Vậy `Q=c'²+s'²` **luôn bị chặn dưới bởi 1 hằng số dương cụ thể** (≈0.086), không thể tiến về 0 — `ε` luôn thực sự "không đáng kể" so với mẫu số, và claim `(\cos\hat\varphi')^2+(\sin\hat\varphi')^2\to1` khi `ε\to0` giờ **đúng vô điều kiện** (không còn phụ thuộc `(Δc,Δs)` học ra là bao nhiêu, vì đã bị `tanh×k` chặn biên độ trước khi cộng vào). Đây là cái giá đánh đổi: `k=0.5` giới hạn Stage A chỉ sửa được correction có biên độ vừa phải (không sửa được lỗi phase quá lớn) — hợp lý vì Stage A chỉ nhắm vào sai số *tinh chỉnh*, không phải dự đoán lại phase từ đầu.
\[
\hat{S}' = \hat{M}\cdot(\cos\hat\varphi' + i\sin\hat\varphi')
\qquad
x_0 = iSTFT(\hat{S}')
\qquad
x = x_0 + R_B(x_0, h)
\]

trong đó \(\varphi_A\) là **mạng con** (Conv2D) bên trong Stage A tạo ra correction thô `(δc,δs)` — bản thân "Stage A" (tên gọi cho cả khối chức năng) gồm \(\varphi_A\) **cộng thêm** bước bound `tanh×k` + cộng vào `(cos φ̂,sin φ̂)` + re-project về unit circle (3 bước còn lại ở trên, không thuộc \(\varphi_A\)). Toàn bộ Stage A hoạt động trên `(cos,sin)` — khớp đúng representation Vocos's `ISTFTHead` đã tính sẵn ở `heads.py`, không cần thêm bước chuyển đổi định dạng nào. \(R_B\) là Stage B (waveform refiner, miền thời gian). Hai stage giải quyết đúng 2 loại lỗi khác nhau đã biết: Stage A nhắm vào phase theo đúng representation wrap-safe của paper/Vocos, Stage B nhắm vào harmonic/transient miền thời gian (đúng thế mạnh đã biết của HiFi-GAN/Snake).

> **Lưu ý đã bỏ thuật ngữ "phase-diff"** dùng ở bản trước — bản đó mô tả không khớp với công thức (công thức dùng phase tuyệt đối, không phải hiệu số), gây mâu thuẫn giữa hình vẽ và công thức. Bản này thống nhất dùng `(cos,sin)` xuyên suốt — vừa wrap-safe, vừa khớp sẵn với code hiện có (`heads.py`'s `cos_p, sin_p`), không cần định nghĩa thêm 1 representation "phase-difference" riêng.

### 3.2. Vì sao tách 2 stage thay vì 1 refiner duy nhất

- Nếu chỉ có Stage B (bản gốc): đặt cược refiner miền thời gian tự suy ra được correction miền tần số chỉ từ waveform đã collapse — không có tiền lệ trực tiếp chứng minh khả thi.
- Nếu chỉ có Stage A: không có cơ chế sửa lỗi harmonic/transient thuần miền thời gian (ví dụ răng cưa ở vùng chuyển tiếp voiced/unvoiced) mà phase-refinement không chạm tới.
- Chi phí Stage A dự kiến rẻ — paper đo được 37.1K cho kỹ thuật *phase-difference + Conv2D* của họ, nhưng đó là thiết kế khác (Stage A ở đây dùng cos/sin correction, kiến trúc/input-output shape khác, xem §3.1) nên **37.1K không phải số đo trực tiếp cho Stage A này**, chỉ là căn cứ để tin rằng 1 module Conv2D nhỏ nhắm đúng vào phase là khả thi ở quy mô rất nhỏ. Ngân sách 40–80K ở §6 là ước tính riêng, cần đo lại thật khi implement, không lấy 37.1K làm số cam kết.

---

## 4. Periodic Waveform Refiner (Stage B) — **đã sửa (điểm cấn #2)**

### 4.1. Vấn đề bản gốc

Block "MRF" trong bản gốc chỉ đổi **kernel size** (3/5/7) giữa các nhánh DWConv song song rồi concat+fuse. Nhưng sức mạnh thật của HiFi-GAN MRF đến từ **dilation tăng dần bên trong mỗi resblock** (thường 1, 3, 5) kết hợp với kernel size — dilation mới là cơ chế chính giúp receptive field lớn với ít tham số, không phải chỉ đổi kernel size. Thiếu trục dilation, claim "mượn sức mạnh MRF của HiFi-GAN" chưa đúng bản chất.

### 4.2. Fix — thêm dilation

```text
                         Input
                           │
             ┌─────────────┼──────────────┐
             │             │              │
             ▼             ▼              ▼
      DWConv k=3      DWConv k=3     DWConv k=3
      dilation=1      dilation=3     dilation=5
             │             │              │
          SnakeBeta     SnakeBeta      SnakeBeta
             │             │              │
             └─────────────┼──────────────┘
                           │
                       Concatenate
                           │
                           ▼
                     1×1 Conv Fusion
                           │
                           ▼
                        Residual Add
```

Giữ kernel size cố định (k=3) nhưng thay đổi dilation (1/3/5) — chi phí tham số gần như không đổi so với bản không-dilation (DWConv không phụ thuộc dilation về số tham số, chỉ phụ thuộc kernel size × channels).

**Tính chính xác receptive field (RF) — sửa lại claim "tăng theo cấp số nhân" ở bản trước, không đúng với đúng kiến trúc đã vẽ**: 3 nhánh trong hình chạy **song song** (không xếp chồng tuần tự), nên RF của cả block là **max** của 3 nhánh, không phải tích/lũy thừa. Với kernel k=3, RF 1 nhánh dilation \(d\) là \((k-1)d+1 = 2d+1\):

\[
\text{RF}_{d=1}=3,\quad \text{RF}_{d=3}=7,\quad \text{RF}_{d=5}=11
\qquad\Rightarrow\qquad \text{RF}_{\text{block}} = \max(3,7,11) = 11
\]

Nếu Stage B xếp \(N\) block như vậy tuần tự (mỗi block có residual riêng, như hình đã vẽ), RF tổng tăng **tuyến tính theo \(N\)** — reach (bán kính ảnh hưởng) mỗi phía của 1 block là \((11-1)/2=5\); khi compose \(N\) block, reach 2 phía cộng dồn tuyến tính: \(5N\) mỗi phía, tức RF tổng \(=2(5N)+1 = 1+10N\) (đẳng thức đúng, **không phải "≈"** — sửa 1 chỗ viết thiếu chặt ở bản trước: `2(5N)+1` và `1+10N` là cùng 1 biểu thức, khai triển đại số thuần túy không có sai số nào cần xấp xỉ), **không phải cấp số nhân**.

**Vì sao công thức cộng dồn này đúng tuyệt đối, không phải xấp xỉ**: với định nghĩa chuẩn của receptive field (tập input xa nhất có thể ảnh hưởng tới 1 output, qua bất kỳ đường tính toán nào), residual connection trong mỗi block **không** làm tăng RF vượt quá nhánh conv — vì nhánh skip (identity) có RF=1, luôn nhỏ hơn nhánh conv (RF=11), nên đường xa nhất luôn là đường đi qua nhánh conv ở **mọi** block. Do đó RF của \(N\) block xếp chồng đúng bằng RF của \(N\) lớp conv nối tiếp thuần túy (không residual) có cùng kernel/dilation — công thức cộng dồn tuyến tính chuẩn (`RF=1+\sum(R_i-1)`) áp dụng nguyên vẹn, không bị nhiễu bởi residual. Muốn tăng theo cấp số nhân thật (kiểu WaveNet) cần **nhân dilation lên qua từng block** (block thứ \(i\) dùng dilation \(1{\times}2^i, 3{\times}2^i, 5{\times}2^i\) thay vì lặp lại đúng 1/3/5 ở mọi block) — đây là hướng mở rộng khả dĩ cho Phase 5 (§13) nếu Phase 2-4 cho thấy cần RF lớn hơn mà không muốn tăng số block tuyến tính.

---

## 5. Kích hoạt tuần hoàn (SnakeBeta) — **đã sửa (điểm cấn #3)**

### 5.1. Công thức — **sửa lại cho đúng (đối chiếu code thật)**

Bản trước viết nhầm công thức của **Snake gốc** (1 tham số) trong khi tài liệu gọi tên "SnakeBeta" xuyên suốt. Đối chiếu `banhmi_train/BigVGan/activations.py` (class `SnakeBeta`, đang dùng thật trong dự án, khớp cấu hình BigVGAN release), công thức đúng là:

\[
SnakeBeta(x) = x + \frac{1}{\beta}\sin^2(\alpha x)
\]

với **`α` (tần số) và `β` (nghịch đảo biên độ) là 2 tham số học riêng biệt theo từng channel** — không dùng chung 1 tham số như Snake gốc. Đây là lý do đặt tên "Beta": tách vai trò tần số và biên độ ra khỏi nhau, đúng bản BigVGAN release thật đang dùng trong `Stage B`.

### 5.2. Bằng chứng tiêu cực đã có sẵn — bổ sung (bản gốc bỏ sót)

Ablation chính thức của paper Vocos gốc (arXiv 2306.00814) đã thử Snake **ngay trong backbone Vocos** và kết luận: **"không cải thiện, thậm chí giảm nhẹ hiệu năng."** Bản gốc liệt "Snake không cải thiện" như một rủi ro *chưa biết* (Risk 2) — thực ra đã có tiền lệ âm tính.

**Quan trọng**: kết quả âm tính đó ở trong ngữ cảnh khác — Snake làm activation *bên trong backbone Vocos* (dự đoán STFT), không phải làm activation trong một *refiner miền thời gian riêng biệt* (đúng vai trò gốc của Snake trong HiFi-GAN/BigVGAN — sửa waveform, không dự đoán phổ). Nên không tự động phủ định MarGan, nhưng phải đưa vào đánh giá rủi ro tường minh thay vì bỏ qua.

---

## 6. Ngân sách tham số (cập nhật theo kiến trúc 2-stage)

**Cam kết tường minh — `n_fft = 1024`, không giảm xuống 512.** Ablation §1 đã chứng minh giảm `n_fft` (1024→512) làm loss_mel **tệ hơn** ở cả 2 mức `dim` (2.20→3.63 và 6.00→7.58) — giảm độ phân giải tần số mất thông tin thật, không phải giảm độ khó dự đoán như tưởng ban đầu. MarGan giữ nguyên `n_fft=1024` cho backbone/head; toàn bộ ngân sách dưới đây tính ở `n_fft=1024`, không phải 512:

| Thành phần | Tham số mục tiêu (tại n_fft=1024) |
|---|---:|
| Lightweight Vocos backbone | 700–900K |
| Complex STFT head (n_fft=1024) | 100–200K |
| **Stage A — Phase refiner (mới)** | **40–80K** |
| Stage B — Periodic waveform refiner | 300–500K |
| **Tổng** | **1.14–1.68M** |

Ngân sách tổng gần như không đổi so với bản gốc (Stage A rất rẻ) — thêm được 1 lớp sửa lỗi đúng bằng chứng đã có mà gần như không tốn thêm ngân sách.

---

## 7. HiFi-GAN — ý tưởng dùng vs. không dùng

### Dùng

- Multi-receptive-field (nay có cả dilation, không chỉ kernel size — §4.2)
- Adversarial waveform training
- Periodic/discriminative modeling (SnakeBeta trong Stage B)
- Multi-scale waveform evaluation (discriminator)

### Không dùng trực tiếp

- Generator HiFi-GAN đầy đủ
- Chuỗi upsample lớn
- Resblock kích thước đầy đủ
- Channel width lớn

---

## 8. Discriminator

Không cần tuân theo ràng buộc 1-2M của generator suy luận. Cấu hình khởi điểm: MPD + MRD chuẩn (giống setup hiện có trong `banhmi_train/Vocos/discriminators.py`), để cô lập đóng góp của generator trước, đúng nguyên tắc thực nghiệm đã dùng xuyên suốt dự án này.

---

## 9. Hàm loss

\[
L = \lambda_{mel}L_{mel} + \lambda_{STFT}L_{STFT} + \lambda_{wave}L_{wave} + \lambda_{adv}L_{adv} + \lambda_{fm}L_{FM} + \lambda_{\varphi}L_{\varphi}
\]

Thêm \(L_\varphi\) — loss riêng cho Stage A. **Định nghĩa chính xác (đã chứng minh, không phải "tương đương" hand-wave)**:

\[
L_\varphi = \mathbb{E}\Bigl[(\cos\varphi' - \cos\varphi_{gt})^2 + (\sin\varphi' - \sin\varphi_{gt})^2\Bigr]
\]

Khai triển bình phương và dùng danh tính \(\cos(A-B)=\cos A\cos B+\sin A\sin B\):

\[
(\cos\varphi'-\cos\varphi_{gt})^2+(\sin\varphi'-\sin\varphi_{gt})^2
= \underbrace{(\cos^2\varphi'+\sin^2\varphi')}_{=1} + \underbrace{(\cos^2\varphi_{gt}+\sin^2\varphi_{gt})}_{=1} - 2(\cos\varphi'\cos\varphi_{gt}+\sin\varphi'\sin\varphi_{gt})
\]
\[
= 2 - 2\cos(\varphi'-\varphi_{gt}) = 2\bigl(1-\cos(\varphi'-\varphi_{gt})\bigr)
\]

Vậy \(L_\varphi\) (MSE trên cặp thô `(c,s):=(\cos\hat\varphi',\sin\hat\varphi')` Stage A thực sự xuất ra) **bằng** \(2(1-\cos(\varphi'-\varphi_{gt}))\) — không phải "tương đương" mơ hồ, mà là **đẳng thức đại số**, với 1 điều kiện cần xét lại chặt chẽ (xem khung sửa ngay dưới, đã tự sửa 1 lượt vì cách phát biểu điều kiện ở lượt trước cũng chưa chuẩn): đẳng thức trên coi `(c,s)` là `(cosφ',sinφ')` đúng nghĩa của 1 góc — tức `c^2+s^2=1` đúng tuyệt đối. Điều này **đúng tuyệt đối cho `φ_gt`** (phase thật, không qua Stage A, luôn nằm đúng trên unit circle) nhưng **không đúng cho output thật của Stage A** — `c,s` qua re-projection có `ε` ở §3.1 có `c^2+s^2=x<1` (không phải "gần đúng 1 nên coi như đúng", mà là 1 giá trị cụ thể nhỏ hơn 1 nghiêm ngặt, chặn dưới bởi §3.1). Đây là hàm khoảng cách góc chuẩn (bằng 0 khi \(\varphi'=\varphi_{gt}\), tăng đơn điệu theo \(|\varphi'-\varphi_{gt}|\) trong khoảng \((-\pi,\pi]\), tự động tuần hoàn đúng chu kỳ \(2\pi\) nhờ `cos`) — **tự động wrap-safe vì hoạt động trên tọa độ Euclid của đường tròn đơn vị, không phải trên góc**, nên không cần công thức "wrapping-aware" riêng biệt như cách paper phải xử lý khi làm việc trực tiếp trên góc/hiệu góc.

**Sửa lại — bản sửa ở lượt rà soát trước tự nó chưa chặt.** Lượt trước bảo lưu bước khai triển `\cos\varphi'\cos\varphi_{gt}+\sin\varphi'\sin\varphi_{gt}=\cos(\varphi'-\varphi_{gt})` rồi chỉ trừ hao đúng phần `1-(\cos^2\varphi'+\sin^2\varphi')` — nhưng danh tính lượng giác đó **chỉ đúng nếu `(\cosφ',\sinφ')` là cos/sin thật của 1 góc thực** (tức unit-norm đúng nghĩa). Output thật của Stage A là cặp `(c,s):=(\cos\hat\varphi',\sin\hat\varphi')` với `c^2+s^2=x:=Q/(Q+\epsilon)<1` (không đúng 1) — áp thẳng danh tích lên `(c,s)` chưa chuẩn hóa như vậy là **chưa hợp lệ**, không chỉ đơn thuần "gần đúng". Cần định nghĩa lại `φ'` cho chặt trước khi dùng danh tích lượng giác.

**Định nghĩa chặt**: đặt `\varphi' := \operatorname{atan2}(s,c)` — hướng thật của vector `(c,s)` (định nghĩa `atan2` áp dụng cho mọi vector khác 0). Theo đúng định nghĩa này, `c=\sqrt{x}\cos\varphi'`, `s=\sqrt{x}\sin\varphi'` là **đẳng thức đúng tuyệt đối** (không xấp xỉ). Khai triển lại `L_\varphi` bằng `x` và `φ'` này:

\[
L_\varphi = (c^2+s^2) - 2(c\cos\varphi_{gt}+s\sin\varphi_{gt}) + 1 = x - 2\sqrt{x}\cos(\varphi'-\varphi_{gt}) + 1
\]

Đây là đẳng thức **đúng tuyệt đối** với `x`,`φ'` đúng nghĩa trên — không phụ thuộc giả định nào. Khi `x=1` (lý tưởng), công thức rút gọn chính xác về \(2(1-\cos(\varphi'-\varphi_{gt}))\) như đã dẫn ở trên.

**Chặn sai số cho `x=Q/(Q+\epsilon)<1` thật**: đặt `\delta:=1-x=\epsilon/(Q+\epsilon)\leq1.165\times10^{-7}` (dùng đúng chặn `Q\geq0.0858` đã chứng minh ở §3.1). Cần chặn `1-\sqrt{x}=1-\sqrt{1-\delta}` — với mọi `\delta\in[0,1]`, có **2 chặn đúng tuyệt đối** (bình phương 2 vế, không dùng xấp xỉ Taylor):

\[
\frac{\delta}{2} \;\leq\; 1-\sqrt{1-\delta} \;\leq\; \delta
\]

(chặn trên: `\sqrt{1-\delta}\geq1-\delta \Leftrightarrow 1-\delta\geq(1-\delta)^2 \Leftrightarrow \delta(1-\delta)\geq0`, đúng với `\delta\in[0,1]`; chặn dưới: `1-\delta/2\geq\sqrt{1-\delta} \Leftrightarrow (1-\delta/2)^2\geq1-\delta \Leftrightarrow \delta^2/4\geq0`, luôn đúng). Từ đó:

\[
L_\varphi - 2\bigl(1-\cos(\varphi'-\varphi_{gt})\bigr) = -\delta + 2\cos(\varphi'-\varphi_{gt})\cdot(1-\sqrt{x})
\]

Dùng `|\cos(\cdot)|\leq1` và `0\leq1-\sqrt{x}\leq\delta`:

\[
\bigl|L_\varphi - 2(1-\cos(\varphi'-\varphi_{gt}))\bigr| \;\leq\; \delta + 2\delta = 3\delta \;\leq\; 3\times1.165\times10^{-7} \;\approx\; 3.5\times10^{-7}
\]

**Kết luận đã sửa lại**: sai số là sai số **tuyệt đối** (không phải "tương đối" như bản sửa trước ghi nhầm), chặn trên đúng là `3\delta\lesssim3.5\times10^{-7}` — gấp ~3 lần con số `≈1.2×10⁻⁷` ghi sai ở lượt sửa ngay trước (con số đó dựa trên 1 bước khai triển không hợp lệ về mặt đại số, dù kết luận định tính "không đáng kể" tình cờ vẫn đúng). Vẫn **nhỏ tới mức vô nghĩa** với mọi mục đích thực tế — nhỏ hơn cả sai số làm tròn float32 điển hình — kết luận cuối cùng (dùng `2(1-\cos\Delta\varphi)` làm dạng đóng cho `L_\varphi` là an toàn) không đổi, chỉ hệ số/cách chứng minh của chặn trên được sửa cho đúng.

**Lưu ý**: đẳng thức trên chỉ đúng cho **L2 bình phương** (MSE, như viết ở trên). Nếu dùng **L1** (`|cosφ'−cosφ_gt| + |sinφ'−sinφ_gt|`) thay vì L2², đó là 1 hàm loss khác — vẫn hợp lệ về mặt wrap-safe (cùng lý do: hoạt động trên tọa độ Euclid), nhưng **không** bằng \(2(1-\cos\Delta\varphi)\) hay bất kỳ hằng số nhân nào của nó — không nên gọi 2 lựa chọn này là "tương đương", chỉ nên nói cả 2 đều wrap-safe, khác nhau ở đặc tính gradient/độ nhạy outlier (L1 cứng hơn với outlier, L2² mượt hơn, chuẩn MSE dễ tune hơn). Mặc định dùng L2² (MSE) vì có dạng đóng đẹp ở trên.

Đây là bổ sung cần thiết mà bản gốc không có (bản gốc chỉ có loss ở mức waveform/mel, không có loss riêng cho Stage A mới).

**Cập nhật sau thực nghiệm Phase 3 (§11.1, Rủi ro 7 ở §12) — `λ_φ` mặc định nên là 0.** Toàn bộ suy luận đại số ở trên (`L_φ` là hàm khoảng cách góc hợp lệ, wrap-safe, có dạng đóng đẹp) vẫn đúng — vấn đề không nằm ở toán học của `L_φ`, mà ở việc **thêm nó vào loss huấn luyện không tạo ra tín hiệu học được**: overfit-test với `λ_φ·L_φ` (λ_φ=10) trong loss cho thấy `L_φ` gần như đứng yên (1.96→1.96 xu hướng) sau 800 bước dù có gradient trực tiếp, trong khi mel_loss vẫn cải thiện tốt qua con đường khác. Xem Rủi ro 7 để biết diễn giải và khuyến nghị (`λ_φ=0` mặc định, giữ `L_φ` làm metric theo dõi phụ, không phải mục tiêu tối ưu).

### Harmonic-Aware Loss (giai đoạn 2, chưa cụ thể hóa) — **đã làm rõ (điểm cấn #4)**

Bản gốc liệt 5 lựa chọn cho \(H(\cdot)\) mà không chọn cụ thể, chưa implement được. Giữ nguyên định hướng thử nghiệm giai đoạn 2, nhưng ghi rõ: **cần chọn 1 biểu diễn cụ thể và code được trước khi đưa vào Phase 2** (ví dụ: harmonic energy qua comb-filter cố định ở các bội số F0, hoặc log-magnitude tại các harmonic bin từ `pyworld`'s F0 estimate — không dùng "instantaneous-frequency-related features" mơ hồ). Không phải phần bắt buộc cho baseline đầu tiên.

---

## 10. Giả thuyết novelty chính

> **Một vocoder Vocos siêu nhẹ, kết hợp (a) phase-refinement nhỏ ở miền tần số ngay trước `iSTFT` và (b) periodic waveform-refinement ở miền thời gian sau `iSTFT`, phục hồi được phần lớn khoảng cách chất lượng với Vocos đầy đủ, trong ngân sách ~1.5M tham số — mỗi stage nhắm đúng loại lỗi mà bằng chứng thực nghiệm/tài liệu đã xác định (Stage A: phase, theo paper; Stage B: harmonic/transient miền thời gian, theo thế mạnh đã biết của HiFi-GAN/Snake).**

3 đóng góp cốt lõi:

1. **Stage A — Phase refinement đúng miền, đúng bằng chứng**: không đặt cược vào refiner miền thời gian tự suy ra correction miền tần số.
2. **Stage B — Periodic MRF có dilation thật**: đúng cơ chế HiFi-GAN gốc, không chỉ đổi kernel size.
3. **Ngân sách tham số tối ưu**: tận dụng chi phí cực rẻ của Stage A (~40-80K) để thêm 1 lớp sửa lỗi mà không phải đánh đổi đáng kể ngân sách dành cho backbone/Stage B.

---

## 11. Kế hoạch thực nghiệm

### 11.1. Baseline — **đã làm rõ (điểm cấn #3)**

**Vấn đề bản trước**: bảng ablation dùng chung nhãn "Vocos nhẹ" cho mọi hàng, nhưng §6 đặt mục tiêu backbone MarGan (700–900K) **nhỏ hơn hẳn** `vocos_small` đã test (dim=160, ~1.99–2.0M đo qua `compare_generators.py`) — nếu Model A tái dùng số đo của `vocos_small` mà Model B-E lại dùng 1 backbone nhỏ hơn chưa từng đo, các hàng trong bảng không còn so sánh cùng 1 biến "Vocos nhẹ" nữa.

**Fix**: tách rõ 2 baseline khác nhau, không gộp chung:

1. `vocos_small` (dim=160/480/8, n_fft=1024) — **đã đo**: loss_mel=6.00, ~1.99M param (`compare_generators.py --kinds vocos_small`). Dùng làm mốc tham chiếu "Vocos nhẹ đã biết", không phải backbone MarGan.
2. **Backbone MarGan riêng** (mục tiêu 700–900K, nhỏ hơn `vocos_small`) — **đã đo** (`compare_generators.py --kinds vocos_dim96,vocos_dim112,vocos_dim128`, n_fft=1024, intermediate_dim=3×dim, num_layers=8, cùng tỉ lệ `vocos_small`):

   | dim | params (backbone+head) | loss_mel |
   |---:|---:|---:|
   | 96 | 898,626 | 8.53 |
   | 112 | 1,134,242 | 7.58 |
   | 128 | 1,394,434 | 7.18 |

   Chỉ **dim=96 (898,626 params)** nằm trong khoảng mục tiêu 800K–1.1M của Model A; dim=112 đã vượt nhẹ trần, dim=128 vượt hẳn (gần chạm tổng ngân sách MarGan đầy đủ 1.14–1.68M, không còn chỗ cho Stage A/B). **Chốt dim=96 làm backbone MarGan cho toàn bộ §11.2 trở đi.** loss_mel=8.53 tệ hơn `vocos_small` (6.00) ~42% — đúng như dự đoán ở Rủi ro 5, đây chính là khoảng cách Stage A/B phải bù lại; chưa tách được đây là lỗi magnitude hay phase (cần đo riêng theo đúng mitigation của Rủi ro 5 trước khi thêm Stage A/B).

### 11.2. Bảng ablation

Tất cả các hàng B–E dùng **backbone MarGan riêng, dim=96 đã chốt ở mục 11.1.2** (898,626 params, không phải dim=160 của `vocos_small`):

| Model | Backbone MarGan | Stage A (phase) | Stage B (waveform) | Dilation MRF | Params (mục tiêu) |
|---|---:|---:|---:|---:|---:|
| A | ✓ | | | | **898,626** *(đã đo, dim=96)* |
| B | ✓ | | ✓ | | ~1.20M–1.40M |
| C | ✓ | | ✓ | ✓ | ~1.20M–1.40M *(= B, xem dưới)* |
| D | ✓ | ✓ | | | ~0.94M–0.98M |
| E (MarGan) | ✓ | ✓ | ✓ | ✓ | ~1.24M–1.48M |

Các cột Params B–D **tính trực tiếp từ số đo A + ngân sách mục tiêu §6**, không phải số mới: A = 898,626 (đã đo, cố định — không còn là khoảng ước tính); B = A + Stage B (300–500K) = 1.20M–1.40M; D = A + Stage A (40–80K) = 0.94M–0.98M; E = A + Stage A + Stage B = 1.24M–1.48M — **khoảng E này hẹp hơn và nằm gọn trong khoảng ước tính ban đầu của §6 (1.14–1.68M)**, vì A giờ là 1 số cố định (898,626, gần trần 900K của khoảng backbone gốc) thay vì cả 1 khoảng. **C = B tuyệt đối, không lệch dù chỉ 1 tham số** — vì §4.2 đã chứng minh dilation không đổi số tham số của DWConv (chỉ phụ thuộc kernel size × channels, không phụ thuộc dilation); cột "Dilation MRF" chỉ đổi RF/chất lượng, không đổi ngân sách, nên khoảng Params của C phải trùng khít khoảng của B — 2 hàng này tồn tại để so sánh `loss_mel`/MOS ở **cùng 1 ngân sách tham số**, cô lập đúng 1 biến (có/không dilation).

`vocos_small` (loss_mel=6.00, ~2.0M) đóng vai trò **mốc so sánh phụ** — nếu Model A (backbone nhỏ hơn, chưa refine) đã tệ hơn hẳn `vocos_small` dù ít tham số hơn, đó là dấu hiệu sớm cho Rủi ro 5 (backbone quá nhỏ, cả magnitude cũng hỏng) trước khi tốn công thêm Stage A/B.

Mỗi hàng cần chứng minh đóng góp độc lập — đúng tinh thần "không chỉ kết hợp 3 kiến trúc có sẵn" đã tự đặt ra trong Risk 1.

### 11.3. Kiểm chứng nhanh trước khi cam kết train full

Theo đúng phương pháp đã dùng cho ablation `dim`×`n_fft` (`test_function/compare_generators.py`, overfit ~800 bước, CPU, vài phút): implement bản MarGan tối giản, chạy overfit test so `loss_mel` trực tiếp với 4 kết quả Vocos đã có trước khi đầu tư train full-scale (có thể mất nhiều ngày cho GAN training thật).

---

## 12. Rủi ro

### Rủi ro 1 — Chỉ là ghép 3 kiến trúc có sẵn
**Giảm thiểu**: Stage A+B nhắm đúng 2 loại lỗi khác nhau có bằng chứng riêng, không phải ghép ngẫu nhiên.

### Rủi ro 2 — Snake không cải thiện
**Đã có tiền lệ âm tính** (Snake trong backbone Vocos, xem §5.2) — nhưng khác ngữ cảnh (refiner miền thời gian riêng, đúng vai trò gốc của Snake). Cần ablation activation (ReLU/LeakyReLU/GELU/Snake/SnakeBeta) trong chính Stage B để xác nhận, không giả định sẽ có tác dụng chỉ vì đã hiệu quả trong HiFi-GAN.

### Rủi ro 3 — Stage B chạy ở sample rate đầy đủ, không phải frame rate (nguyên nhân gốc, không chỉ "tốn latency" chung chung)

**Sửa lại toàn bộ so sánh này — bản trước quên mất Stage A là Conv2D trên lưới `(F×T_frame)`, không phải chỉ chạy theo 1 trục thời gian.** §3.1 đã nói rõ Stage A là "Conv2D nhỏ" trên `(M̂,cosφ̂,sinφ̂)` — dữ liệu này có 2 trục: tần số (`F=n_fft/2+1=513` bin) **và** thời gian (`T_frame`, ví dụ ~86 với hop=256). FLOPs của Conv2D tỉ lệ với **số vị trí lưới `F×T_frame`**, không phải chỉ `T_frame` — bản trước ngầm coi Stage A "rẻ vì chạy ở frame rate" như thể nó là Conv1D, bỏ sót hẳn thừa số `F`.

**Công thức tổng quát (đúng cho mọi loại conv — 1D/2D, dense/depthwise)**: với 1 layer có `N` = số vị trí output (grid size — là `T` cho Conv1D, `F×T` cho Conv2D), `Params` = số tham số của layer:

\[
\text{FLOPs}_{\text{layer}} \approx 2N\cdot\text{Params}_{\text{layer}}
\]

(mỗi tham số được dùng lại đúng 1 lần cho mỗi vị trí output — đúng bản chất "weight sharing" của convolution, không phụ thuộc 1D/2D hay dense/depthwise). Cộng tuyến tính qua các layer trong 1 stage (đã chứng minh trước đó) — **với 1 điều kiện cần nói rõ, vì chính chỗ này đã từng là nguồn gốc lỗi ở lượt sửa trước**: bước gộp `FLOPs_stage ≈ 2N·Params_stage` (đưa `N` ra ngoài tổng, thay vì `Σ_layer 2N_layer·Params_layer`) chỉ hợp lệ nếu **mọi layer trong stage dùng chung 1 grid size `N`** — tức stage không có layer nào strided/pooling làm đổi kích thước lưới. Điều này đúng theo đúng thiết kế đã chốt ở §3.1: Stage A xuất `(cosφ̂',sinφ̂')` **cùng shape** `(F×T_frame)` với input `(cosφ̂,sinφ̂)` (không đổi kích thước lưới), Stage B xuất residual `r` **cùng độ dài** `T_sample` với `x_0` — cả 2 đều là refiner giữ nguyên shape theo đúng định nghĩa ở §3.1/§4, không phải mạng có downsample/upsample nội bộ. Nên `N` không đổi qua các layer trong cùng 1 stage, và bước gộp là chính xác, không phải xấp xỉ thêm ẩn.

\[
\text{FLOPs}_{\text{Stage A}} \approx 2(F\cdot T_{frame})\cdot\text{Params}_A
\qquad
\text{FLOPs}_{\text{Stage B}} \approx 2(256\,T_{frame})\cdot\text{Params}_B
\]

(dùng `T_sample = hop\_length\cdot T_{frame} = 256\,T_{frame}` cho Stage B). Tỉ số:

\[
\frac{\text{FLOPs}_B}{\text{FLOPs}_A} = \frac{256\cdot\text{Params}_B}{F\cdot\text{Params}_A} = \frac{256}{513}\cdot\frac{\text{Params}_B}{\text{Params}_A} \approx 0.499\cdot\frac{\text{Params}_B}{\text{Params}_A}
\]

**Thay đúng ngân sách mục tiêu ở §6** (`Params_A∈[40K,80K]`, `Params_B∈[300K,500K]`), tính 2 đầu mút:

\[
\frac{\text{FLOPs}_B}{\text{FLOPs}_A}\Big|_{\min} = 0.499\times\frac{300K}{80K} \approx 1.87
\qquad
\frac{\text{FLOPs}_B}{\text{FLOPs}_A}\Big|_{\max} = 0.499\times\frac{500K}{40K} \approx 6.24
\]

**Kết luận đã sửa**: Stage B tốn FLOPs cao hơn Stage A khoảng **~1.9× đến ~6.2×** (tùy vị trí trong khoảng ngân sách mục tiêu) — **không phải 256× như bản trước khẳng định sai**. Thừa số `256` (từ `hop_length`) và thừa số `1/513` (từ `F` của Stage A, gần như triệt tiêu hẳn thừa số 256 — vì `256/513≈0.5`) gần như **bù trừ lẫn nhau**; chênh lệch FLOPs thật giữa 2 stage chủ yếu đến từ **tỉ lệ ngân sách tham số** (`Params_B/Params_A`, khoảng 3.75–12.5×) chứ không phải từ khác biệt frame-rate/sample-rate như tưởng ban đầu. Rủi ro RTF vẫn có thật (Stage B vẫn đắt hơn Stage A vài lần), nhưng ở mức độ **thấp hơn nhiều bậc** so với con số 256× đã công bố sai ở 2 lượt sửa trước — cần đính chính rõ để không đánh giá quá mức rủi ro này khi quyết định độ nông/sâu của Stage B ở Phase 2/5.

**Giảm thiểu**:
- Đo RTF của riêng Stage B ngay ở Phase 2 (§13) — trước khi cộng thêm Stage A, để biết chính xác chi phí compute ở sample rate đầy đủ, không đợi tới khi ráp đủ pipeline mới phát hiện chậm.
- Giữ Stage B **rất nông** (ít block, không cần nhiều lớp như Stage MRF của HiFi-GAN vốn phải xây receptive field từ đầu) — vì Stage B chỉ sửa lỗi *residual* (đã có `x0` làm nền tốt), không cần receptive field lớn như khi generate từ đầu.
- Channel width giới hạn mạnh hơn mức "bình thường" cho 1 module chạy ở sample rate — ngân sách 300-500K tham số ở §6 nên ưu tiên **ít layer, hẹp**, không phải nhiều layer nông.
- Không nên hạ sample rate Stage B xử lý rồi upsample lại residual (dù về lý thuyết giảm compute) — làm vậy tái tạo lại đúng vấn đề upsampling-stack mà Vocos vốn tránh, đi ngược mục tiêu ban đầu.

### Rủi ro 4 — Residual/phase-correction học ra gần-0 (không cần thiết)
**Giảm thiểu**: đo \(\|r\|\) (Stage B) và \(\|(\Delta_c,\Delta_s)\|\) (Stage A, đo trực tiếp trên correction đã cộng vào `(cos,sin)` trước khi re-project) riêng biệt, so error spectrogram trước/sau mỗi stage để xác định stage nào thực sự đóng góp.

### Rủi ro 5 (mới) — Stage A không đủ sức nếu backbone Vocos quá nhỏ
Nếu Vocos backbone <800K khiến cả **magnitude** cũng sai (không chỉ phase), Stage A (chỉ sửa phase) không giải quyết được vấn đề gốc. **Giảm thiểu**: kiểm tra riêng magnitude loss của backbone trước khi thêm Stage A/B — nếu magnitude đã kém, phải tăng backbone trước, không phải thêm refiner.

### Rủi ro 6 (mới) — Mitigation của Rủi ro 3 (RTF) mâu thuẫn ngầm với mục tiêu MOS của chính Stage B

§2b đặt 3 tiêu chí thành công: param↓, MOS↑, RTF↓. Rủi ro 3 khuyến nghị giữ Stage B **rất nông/hẹp** để giảm chi phí compute ở sample rate đầy đủ — hợp lý cho RTF↓. Nhưng Stage B tồn tại chính là để sửa lỗi harmonic/transient miền thời gian (lý do tách 2 stage ở §3.2, đóng góp cốt lõi #2 ở §10) — nếu ép quá nông/hẹp, nó có thể không còn đủ sức sửa lỗi thật, đe dọa trực tiếp tiêu chí MOS↑. Đây là đánh đổi 2 chiều giữa chính 2 tiêu chí thành công của tài liệu này, **không có câu trả lời lý thuyết** — chỉ Phase 2 (Stage B alone, đo cả `loss_mel`/UTMOS lẫn RTF cùng lúc) mới trả lời được độ nông/hẹp nào vẫn giữ được cả 2. **Giảm thiểu**: nếu Phase 2 cho thấy đánh đổi gay gắt (nông đủ nhanh thì không đủ sửa lỗi), chấp nhận nới RTF cao hơn mục tiêu ban đầu và tìm điểm cân bằng qua Phase 5 (tune ngân sách), thay vì cố ép cả 2 tiêu chí cùng lúc mà không có dữ liệu.

### Rủi ro 7 (mới) — `L_φ` không học được dù có giám sát trực tiếp; giả định ở §9 về vai trò của nó cần xét lại

**Bằng chứng thực nghiệm** (Phase 3, §11.1): overfit-test dim=96+Stage A, 800 bước. Không có `λ_φ·L_φ` trong loss: `L_φ` 1.9989→1.9985 (đứng yên). **Có** `λ_φ·L_φ` trong loss (λ_φ=10, không nhỏ): `L_φ` 1.9637→1.9619 — vẫn gần như đứng yên, dù có gradient trực tiếp từ chính nó. Trong khi đó `mean‖(Δc,Δs)‖≈0.58-0.61` (không nhỏ, so với trần lý thuyết `k√2≈0.707` ở §3.1) — Stage A **vẫn học ra correction thật**, chỉ là correction đó không hội tụ về phase thật (`L_φ` không giảm), và `mel_loss` vẫn cải thiện tốt (8.53→7.86-7.88) qua Stage A dù `L_φ` không cải thiện tương ứng.

**Diễn giải khả dĩ nhất**: phase tuyệt đối của STFT gần như không có cấu trúc dự đoán được ở mức từng bin tần số (entropy cao) — khớp với hiện tượng đã biết trong xử lý audio (và khớp tinh thần cảnh báo của `arXiv 2607.24323` về "phasiness"). Chất lượng cảm nhận/mel-loss phụ thuộc chủ yếu vào magnitude và tính *tự nhất quán tương đối* của phase (không có bậc nhảy/gãy ở biên khung, hài hòa đúng harmonic), không phải khớp tuyệt đối với 1 giá trị phase "đúng" duy nhất — điều này giải thích tại sao Stage A vẫn cải thiện mel_loss (đóng góp thật) mà không cần `L_φ` giảm.

**Hệ quả cho tài liệu này**: giả định ở §9 rằng `L_φ` là 1 loss/metric có ý nghĩa để tối ưu **và** theo dõi tiến độ Stage A **chưa được xác nhận, có bằng chứng ngược lại**. Không phủ định toàn bộ Stage A (mel_loss vẫn cải thiện thật — bằng chứng chính vẫn nên là mel_loss/UTMOS, không phải `L_φ`), nhưng `λ_φ·L_φ` trong hàm loss tổng (§9) **có thể là 1 số hạng vô ích hoặc thậm chí có hại** (kéo gradient về hướng không giúp ích thật, cạnh tranh với mel_loss/adversarial loss thật). **Giảm thiểu**: coi `λ_φ=0` là lựa chọn mặc định an toàn hơn cho Phase 4 trở đi (không thêm `L_φ` vào loss thật) cho tới khi có bằng chứng ngược lại rõ ràng hơn (ví dụ ở quy mô dữ liệu lớn hơn, không chỉ overfit 1 clip); giữ `L_φ`/`‖(Δc,Δs)‖` như **metric theo dõi phụ** (không phải mục tiêu tối ưu) để phát hiện sớm nếu Stage A học ra correction gần-0 (Rủi ro 4), nhưng không dùng nó để đánh giá "Stage A có đang sửa phase đúng không".

---

## 13. Thứ tự phát triển đề xuất

### Phase 1 — Backbone MarGan baseline — **hoàn thành, bao gồm cả kiểm chứng Rủi ro 5**
Đã đo (`compare_generators.py --kinds vocos_dim96,vocos_dim112,vocos_dim128`): dim=96 là điểm duy nhất khớp mục tiêu 800K–1.1M của §6 (898,626 params), loss_mel=8.53 — tệ hơn `vocos_small` (6.00) ~42%.

**Kiểm chứng Rủi ro 5** (đo magnitude loss riêng, hàm `vocos_magnitude_loss()` mới thêm vào `compare_generators.py` — L1 trực tiếp giữa `mag=exp(head.out(...)).clamp(1e2)` và `spec` gốc, bỏ hoàn toàn phase/iSTFT/mel):

| dim | loss_mel | magnitude_l1 (tuyến tính, không nén log) |
|---:|---:|---:|
| 512 (full) | 2.20 | 0.779 |
| 160 (small) | 6.00 | 0.984 |
| **96 (MarGan)** | 8.53 | **1.020** |

loss_mel tệ đi ~3.9× (512→96) nhưng magnitude_l1 chỉ tệ đi ~1.3× — magnitude ở dim=96 không "vỡ" theo cùng tỉ lệ với mel_loss tổng (dù đo trên 2 thang khác nhau — tuyến tính vs log-nén — nên không so trực tiếp % được, chỉ so xu hướng tương đối). **Kết luận Rủi ro 5**: chưa có bằng chứng magnitude đã hỏng nặng ở dim=96 — khoảng cách chất lượng nhiều khả năng tập trung ở phase (đúng tiền đề Stage A), không cần tăng backbone trước khi thử Stage A/B như kịch bản xấu Rủi ro 5 đặt ra. Có thể tiến sang Phase 2/3.

### Phase 2 — Thêm Stage B (không Stage A) — **hoàn thành**
Module thật ở `banhmi_train/Vocos/stage_b.py` (channels=224, num_blocks=2, đúng sơ đồ §4 — 3 nhánh DWConv dilation 1/3/5 + SnakeBeta + fuse 1×1 + residual). **Lưu ý phát hiện khi implement**: công thức `R_B(x_0, h)` ở §3.1 ngụ ý Stage B nhận cả `h` (latent gốc), nhưng sơ đồ §4 chỉ vẽ 1 input `x0` — 2 chỗ chưa khớp nhau. Đã chọn theo đúng sơ đồ §4 (bản chi tiết kiến trúc), chỉ dùng `x0`, **không** âm thầm quyết — ghi rõ trong code là còn bỏ ngỏ việc điều kiện hoá `h` (muốn thêm sẽ cần cơ chế đưa `h` từ frame rate lên sample rate của `x0`, chưa có trong thiết kế hiện tại).

Đã đo (`compare_generators.py`, kind `margan_dim96_stageB`):

| Model | Params | loss_mel |
|---:|---:|---:|
| dim=96 backbone alone | 898,626 | 8.53 |
| dim=96 + Stage B | 1,211,555 | **7.51** |

Cải thiện 8.53→7.51 (~12%), lớn hơn Stage A (~7.6%) — hợp lý vì Stage B tốn tham số nhiều hơn (~313K so với ~44K). Đường loss trong lúc train **không mượt như Stage A** (bước 600→700: 7.40→8.42 rồi mới hồi về 7.69 ở bước 800) — không phải lỗi, nhưng là tín hiệu Stage B (waveform thô, sample rate đầy đủ) nhạy với LR cố định (2e-4, dùng chung cho mọi kind trong harness test) hơn Stage A — đáng cân nhắc khi tune thật ở Phase 5, không cần chặn Phase 4.

### Phase 3 — Thêm Stage A (không Stage B) — **hoàn thành (overfit-test + kiểm chứng L_φ, xem Rủi ro 7)**
**Đã đo** (`test_function/compare_generators.py`, kind mới `margan_dim96_stageA` — module thật ở `banhmi_train/Vocos/stage_a.py`, đúng công thức tanh×k+re-project đã chứng minh ở §3.1, chưa wire vào training loop production, chỉ test riêng qua harness overfit như Phase 1):

| Model | Params | loss_mel |
|---:|---:|---:|
| dim=96 backbone alone | 898,626 | 8.53 |
| dim=96 + Stage A | 942,404 | **7.88** |

Stage A cải thiện loss_mel 8.53→7.88 (~7.6%) với +43,778 tham số (~4.9%, đúng khoảng mục tiêu 40-80K ở §6) — bù được ~26% khoảng cách tới `vocos_small` (6.00, tốn gấp 2.2 lần tham số).

**Đo `L_φ` trực tiếp (§9) + `‖(Δc,Δs)‖` (Rủi ro 4) — phát hiện quan trọng, không như kỳ vọng ban đầu.**

| Điều kiện huấn luyện | loss_mel cuối | `L_φ` trước Stage A | `L_φ` sau Stage A | mean `‖(Δc,Δs)‖` |
|---|---:|---:|---:|---:|
| Chỉ mel_loss (không có `L_φ` trong loss) | 7.88 | 1.9989 | 1.9985 | 0.6074 |
| mel_loss + `λ_φ·L_φ` (λ_φ=10, có trong loss) | 7.86 | 1.9637 | 1.9619 | 0.5828 |

`L_φ≈2` tương đương phase gần như **ngẫu nhiên hoàn toàn** so với ground truth (`cos(Δφ)≈0` trung bình; `L_φ=0` mới là khớp hoàn hảo). Dù thêm `λ_φ·L_φ` **trực tiếp vào loss huấn luyện** (không chỉ đo hậu kỳ), `L_φ` chỉ giảm từ 1.99→1.96 sau 800 bước — gần như không học được — trong khi `mean‖(Δc,Δs)‖≈0.58-0.61` cho thấy Stage A **vẫn học ra correction có biên độ đáng kể** (không phải học ra gần-0, không rơi vào lo ngại Rủi ro 4 theo nghĩa "vô dụng"), chỉ là correction đó **không hội tụ về phase thật**.

**Diễn giải**: đây không phải lỗi implementation — khớp với hiện tượng đã biết trong xử lý audio: phase *tuyệt đối* của STFT gần như không có cấu trúc dự đoán được ở mức từng bin (entropy cao), trong khi chất lượng cảm nhận/mel-loss phụ thuộc chủ yếu vào magnitude và tính *tự nhất quán tương đối* của phase, không phải khớp tuyệt đối với phase thật — đúng lý do các vocoder dựa iSTFT (kể cả Vocos gốc) chưa bao giờ thực sự tối ưu để khớp phase tuyệt đối. mel_loss vẫn cải thiện (8.53→7.86-7.88) dù `L_φ` gần như đứng yên — chứng tỏ Stage A đóng góp thật qua 1 con đường khác con đường "khớp phase thật" mà §9 giả định.

**Hệ quả cần ghi nhận tường minh (không giấu)**: giả định ở §9 rằng `L_φ` là 1 loss/metric *có ý nghĩa để tối ưu và theo dõi* cho Stage A **có khả năng sai** — bằng chứng thực nghiệm cho thấy nó gần như không học được dù có giám sát trực tiếp, trong khi mel_loss (thứ thực sự phản ánh chất lượng) vẫn cải thiện qua con đường khác. Xem thêm Rủi ro 7 (mới, §12) về hệ quả này.

### Phase 4 — Kết hợp Stage A + Stage B (MarGan đầy đủ) — **hoàn thành**
Module thật `_MarGan` trong `test_function/compare_generators.py` (kind `margan_dim96_full`) — đúng pipeline §3.1: backbone → head.out → Stage A refine phase → reconstruct complex STFT → iSTFT → `x0` → Stage B residual → `x0+r`.

| Model | Params | loss_mel |
|---:|---:|---:|
| dim=96 alone (Phase 1) | 898,626 | 8.53 |
| dim=96 + Stage A (Phase 3) | 942,404 | 7.88 |
| dim=96 + Stage B (Phase 2) | 1,211,555 | 7.51 |
| **dim=96 + Stage A + Stage B (Phase 4)** | 1,255,333 | **7.47** |
| *(tham chiếu)* `vocos_small` | 1,988,546 | 6.00 |

**Cả 2 stage vẫn đóng góp khi ráp chung** (7.47 thấp hơn cả 2 mốc riêng lẻ 7.51 và 7.88) — không phải 1 stage làm hết việc, đúng yêu cầu "mỗi hàng chứng minh đóng góp độc lập" ở §11.2. Nhưng **đóng góp chồng lấn, không cộng dồn đầy đủ**: nếu 2 stage sửa 2 loại lỗi hoàn toàn tách biệt như §3.2 giả định, cải thiện cộng dồn kỳ vọng là `8.53-0.65-1.02≈6.86`, thực tế chỉ đạt 7.47 — khoảng 0.6 "biến mất" so với kỳ vọng cộng dồn, cho thấy 1 phần lỗi Stage A và Stage B cùng sửa trùng nhau (không hoàn toàn bù 2 loại lỗi độc lập như thiết kế kỳ vọng). Training cũng có đoạn dao động mạnh hơn Stage A một mình (bước 400→500: 11.02→12.20, tăng, rồi mới giảm dần) — cùng xu hướng bất ổn đã thấy ở Stage B một mình (Phase 2), không phải hiện tượng mới.

**Vị trí hiện tại so với mục tiêu §2b**: 1.26M tham số (dưới `baseline` HiFi-GAN ~1.6-2.2M ✓), nhưng loss_mel=7.47 vẫn còn cách `vocos_small` (6.00) khá xa — chưa đủ dữ liệu để nói MOS/UTMOS sẽ ra sao (overfit-test chỉ đo mel_loss trên 1 clip, chưa train GAN thật/chưa đo UTMOS). Bước tiếp theo hợp lý là Phase 5 (tune ngân sách/kiến trúc để thu hẹp khoảng cách với `vocos_small`) trước khi đầu tư Phase 6 (train GAN đầy đủ, tốn nhiều ngày).

### Phase 5 — Tối ưu ngân sách — **bắt đầu, phát hiện quan trọng về LR**
Mục tiêu: \(1.14M \le Params \le 1.68M\) (khớp tổng §6). MarGan đầy đủ ở Phase 4 mới dùng 1.26M/1.68M — còn dư ~420K để thử tăng backbone hoặc Stage B.

**Thử nghiệm đầu tiên: backbone dim=96→112 (giữ nguyên Stage A/B)**, dùng chung `--lr` (mới thêm vào `compare_generators.py` để test được biến này):

| Model | Params | loss_mel (LR=2e-4, mặc định cũ) | loss_mel (LR=1e-4) |
|---|---:|---:|---:|
| dim=96 full | 1,255,333 | 7.47 | 8.10 |
| dim=112 full | 1,490,949 | 8.56 *(tệ hơn — nghi ngờ)* | **7.08** |

**Ở LR=2e-4, dim=112 trông tệ hơn dim=96 — nhưng đây là artifact của bất ổn training, không phải bản chất kiến trúc.** Log LR=2e-4 của dim=112 cho thấy bước 500 đạt loss_mel=6.78 (tốt nhất từng đo trong toàn bộ investigation) rồi tăng vọt lên 9.03→9.74 trước khi hồi về 8.56 ở bước 800 — mất hẳn điểm tốt nhất giữa chừng vì LR quá cao cho model lớn hơn. Chạy lại ở LR=1e-4 (ổn định, cả 2 đường loss mượt, không dao động): **dim=112 thắng rõ dim=96 (7.08 vs 8.10, ~13%)** — đảo ngược đúng kết luận sai ở LR cũ, xác nhận giả thuyết ban đầu của Phase 5 (backbone lớn hơn trong ngân sách cho phép giúp ích thật).

**Lưu ý phương pháp quan trọng**: 2 cột LR trong bảng trên **không so sánh chéo được** — LR thấp hơn hội tụ chậm hơn trong cùng 800 bước cố định (dim=96 tự nó cũng tệ hơn ở LR=1e-4: 8.10 vs 7.47 của chính nó ở LR=2e-4). So sánh công bằng duy nhất là **giữa 2 dim ở cùng 1 LR** — kết luận dùng được: dim=112 > dim=96 ở LR=1e-4. Chưa xác định được liệu 800 bước đã đủ để LR=1e-4 hội tụ hết chưa (có thể cả 2 còn cải thiện thêm nếu train lâu hơn) — cần cẩn trọng khi đem 7.08 so trực tiếp với các mốc Phase 1-4 (đo ở LR=2e-4).

**Việc còn lại của Phase 5**: xác nhận số bước cần thiết để LR=1e-4 hội tụ ổn định (không chỉ tin 800 bước cố định); thử nốt nhánh tăng Stage B (channels/num_blocks) thay vì backbone, so cost-effectiveness giữa 2 hướng; sau đó mới tune dilation/bottleneck ratio như dự kiến ban đầu.

### Phase 6 — Adversarial training đầy đủ
MPD + MRD + feature matching + adversarial loss.

### Phase 7 — Ablation đầy đủ
Chạy toàn bộ bảng §11.2.

---

## 14. Đánh giá

Giữ nguyên nhóm metric đề xuất ở bản gốc (MOS/UTMOS/PESQ/STOI, MCD/spectral convergence, F0 RMSE/correlation, param count/MACs/RTF), bổ sung:

- **\(L_\varphi\) trên `(cos,sin)`** riêng cho Stage A (§9) — wrap-safe tự nhiên nhờ đo trên tọa độ Euclid, không cần công thức wrapping-aware riêng như khi làm việc trực tiếp trên góc.
- **\(\|r\|\) và \(\|(\Delta_c,\Delta_s)\|\) trước/sau mỗi stage** — bắt buộc để trả lời Rủi ro 4.

So sánh quan trọng nhất vẫn là: **MOS/spectral quality vs. tham số và RTF**, đặt cạnh 4 điểm dữ liệu đã có từ ablation `dim`×`n_fft` để thấy MarGan có thực sự phá được giới hạn "dim quyết định gần như tuyệt đối" hay không.
