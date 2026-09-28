# TODO — việc còn tồn đọng

**Quy ước tên gọi**: bản resblock tuần tự (`resblock="mb"`, expansion=1,1,1)
được gọi tắt là **SEQ(1,1,1)** trong mọi tài liệu từ đây trở đi, để phân biệt
rõ với MRF(1,1,1) (song song).

## 1. Train lại SEQ(1,1,1) đủ 1500 epoch — ưu tiên cao

**Lý do**: benchmark nhanh (forward-pass, chưa train, qua T-sweep +
ONNX Runtime 2 luồng) cho thấy bản **tuần tự** (2 khối `ResBlockInverted`
nối tiếp, expansion=1,1,1, nhẹ nhất — 1.107.968 tham số) **nhanh hơn
baseline ở MỌI T đã test (32→700 latent frame, 0.37s→8.13s audio)**,
không có điểm đảo chiều như MRF song song (vốn chậm hơn baseline rõ rệt ở
câu dài, +17.3% ở T=700). Đây là bằng chứng mạnh cho thấy vấn đề RTF của
MRF song song đến từ việc nhân 3 nhánh depthwise-conv cùng lúc (memory-bound
dưới đa luồng), không phải bản thân depthwise conv "xấu" — tuần tự (ít
nhánh hơn) tránh được vấn đề này.

Overfit-test cũ (800 bước, PyTorch) cũng cho thấy tuần tự(1,1,1)
loss_mel=8.9742 tốt hơn nhẹ MRF song song(1,1,1)=9.0788 — không có dấu
hiệu đánh đổi chất lượng.

**Việc cần làm**:
1. Train SEQ(1,1,1) đủ 1500 epoch (giống lịch MRF(1,1,1) đã làm).
2. Dùng đúng `canonical_split.json` (giống hệt tập train/val/test của
   baseline, xây bằng cách load lại checkpoint baseline thật — xem
   `mrf_full_investigation_report.md` §8.2 cho bối cảnh lỗi leakage gốc).
3. Eval đầy đủ: PyTorch (seed cố định) + PTQ + ONNX Runtime/VNNI (seed đã
   biết không kiểm soát được ở tầng ONNX, dùng n=500 để trung bình ổn định).
4. Kiểm định thống kê (Mann-Whitney U hoặc Wilcoxon paired vì cùng test-set
   với baseline) so với baseline, giống MRF(1,1,1) đã làm.
5. Cập nhật `paper_ready_report.md` nếu kết quả tốt hơn MRF(1,1,1) ở RTF mà
   không đánh đổi chất lượng — có thể trở thành kết quả chính thay vì
   MRF(1,1,1).

**Chi phí ước tính**: ~2 ngày training (dựa trên tốc độ MRF(1,1,1) đã đo:
~5.2-5.7 phút/epoch × 1500 epoch), + ~1-2 giờ eval đầy đủ.

**Trạng thái**: đang chạy (lần 5, từ epoch 0, đủ 3 lớp bảo vệ NaN — xem
`docs/nan_collapse_root_cause_and_fix.md`) — `/home/capstone/seq111_final`,
log `/tmp/seq111_final.log`, TensorBoard port 6010, watchdog
`/tmp/seq111_final_watchdog.log`.

Lịch sử NaN (SEQ(1,1,1) là bản đầu tiên trong 3 model từng dính lỗi này khi
chạy không có gradient clip):
- **Lần 1** (2026-09-20): sập NaN vĩnh viễn ở `dp` (StochasticDurationPredictor)
  từ step≈15679 (~epoch 20). Root cause: thiếu epsilon-guard ở 2 phép chia
  trong `banhmi_train/vits/utils/transforms.py::_rational_quadratic_spline`
  (`theta = ... / input_bin_widths` và `delta = heights / widths`) — dưới
  bf16, phép trừ cumsum có thể triệt tiêu số học về đúng 0.0, gây chia 0 →
  NaN, đầu độc vĩnh viễn trạng thái Adam của `dp` (trong khi `dec`/`enc_p`
  không bị ảnh hưởng vì không phụ thuộc đầu ra của `dp`). Đã vá bằng
  `.clamp_min(_EPS)` ở cả 2 chỗ. Archive: `..._PRENAN_BROKEN`.
- **Lần 2** (2026-09-21): dù đã vá lần 1, vẫn sập NaN lại ở `dp` — lần này
  muộn hơn (~epoch 47 thay vì ~epoch 20), cùng chữ ký (loss_mel/loss_kl
  khỏe, `dp` degenerate). `dp` còn được gọi ở chiều reverse ngay trong
  `training forward()` mỗi bước (`synthesizer.py:168`, phục vụ duration
  discriminator VITS2), không chỉ lúc infer — bề mặt lỗi số học rộng hơn 2
  chỗ đã vá, chưa dò hết được từng phép toán. Archive: `..._v2_NaN_ep47`.
- **Quyết định**: thay vì tiếp tục dò từng phép toán, bổ sung
  `--gradient_clip_val 1.0` — đúng kỹ thuật baseline gốc đã dùng để ổn định
  SDP (baseline cũng từng sập NaN không có clip, xem
  `baseline_default_debug_anomaly`, rồi ổn định hẳn từ epoch 89 khi bật
  clip=1.0, chạy khỏe hết 1500). Giữ nguyên 2 chỗ đã vá (vẫn là lỗ hổng
  thật, chỉ chưa đủ một mình) + grad-clip làm lớp phòng thủ thứ 2 chống mọi
  gradient đột biến, không riêng 2 phép chia đã biết.
- **QUAN TRỌNG — gradient clip thực ra làm TỆ HƠN, đã loại bỏ khỏi hướng
  sửa**: kiểm tra trực tiếp TensorBoard scalar của v3 (có clip=1.0) thấy
  `loss_mel` và `loss_kl` cũng NaN vĩnh viễn cùng lúc với `loss_dur` (từ
  step=17849) — khác hẳn v1/v2 (không clip), nơi chỉ `dp` hỏng còn
  `loss_mel`/`loss_kl` vẫn khỏe suốt hàng chục nghìn bước sau đó. Nguyên
  nhân: `clip_grad_norm_` tính 1 norm toàn cục trên MỌI tham số rồi nhân
  cùng 1 hệ số scale cho tất cả — hễ 1 tham số (`dp`) có gradient NaN thì
  norm toàn cục NaN theo, hệ số scale NaN đó lan sang cả `dec`/`enc_p` vốn
  độc lập và trước đây an toàn. Baseline gốc kiểm tra lại cũng cho thấy
  đúng pattern: `baseline_v2_noclip`'s version_0 (chưa bật clip) tự nó
  cũng đã NaN vĩnh viễn ở `dp` từ step≈38949 (~epoch 50) — nhưng vì
  `loss_mel`/`dec` không bị lan NaN (không có clip), checkpoint "best"
  vẫn tiếp tục được lưu bình thường tới epoch 89, và người train trước đó
  đơn giản là **resume từ 1 checkpoint TRƯỚC điểm sập** với clip=1.0 bật
  từ đó — không phải clip "chữa" được sự cố, mà là chọn đúng checkpoint
  sạch để tiếp tục. Kết luận: **không dùng gradient clip nữa** — quay lại
  chỉ dùng 2 epsilon-guard đã vá, và tìm cho ra root cause thật qua
  `--detect_anomaly` (đang chạy) thay vì che bằng clip.
- **Root cause thật, tìm được qua research cộng đồng (coqui-ai/TTS#1959)**:
  cùng 1 lớp lỗi đã được cộng đồng VITS ghi nhận — khi TOÀN BỘ giá trị đầu
  vào của 1 lần gọi `_rational_quadratic_spline` rơi ra ngoài `tail_bound`
  (±5.0, dùng bởi `ConvFlow` trong `dp`), `inside_interval_mask` toàn
  `False` → tensor rỗng được đưa vào hàm spline → undefined behavior
  (Coqui báo crash cứng; ở bf16 có thể thành NaN thay vì crash). Code của
  dự án (`_unconstrained_rational_quadratic_spline` trong `transforms.py`)
  bị đúng lỗ hổng y hệt, không có guard. Đã vá theo đúng pattern Coqui
  dùng: chỉ gọi `_rational_quadratic_spline` khi `inside_interval_mask.any()`
  — khi không có phần tử nào "trong khoảng", bỏ qua an toàn (outputs/
  logabsdet đã được điền đúng cho toàn bộ phần "ngoài khoảng" từ trước).
  Đây là 1 root cause khả dĩ thực sự (khác 2 chỗ epsilon-guard trước, vốn
  chỉ trì hoãn chứ chưa giải quyết dứt điểm). Đang test qua resume tốc độ
  bình thường từ checkpoint sạch cuối (epoch=44, không clip) —
  `/home/capstone/seq111_finder`, log `/tmp/seq111_finder_v2.log`.
- **Phát hiện thêm về baseline**: soi kỹ TensorBoard toàn bộ lịch sử
  baseline mới thấy **baseline KHÔNG thực sự được "fix"** — `version_1`
  (giai đoạn đã bật clip=1.0, resume từ epoch=89) cũng sập NaN toàn bộ 4
  scalar y hệt SEQ, ở step=341699. Chỉ có `version_2` (resume tiếp từ 1
  checkpoint sạch khác ở epoch=870) mới sống sót tới hết 1500 epoch.
  Nghĩa là thành công của baseline chỉ đến từ **retry nhiều lần từ
  checkpoint sạch cho tới khi may mắn vượt qua**, không phải một fix bền
  vững — đúng chiến lược brute-force đang áp dụng cho SEQ, không hơn.
- **Đã kiểm chứng vì sao MRF(1,1,1) không cần clip mà SEQ(1,1,1) cần**:
  ban đầu tưởng do resblock tuần tự kém ổn định hơn song song, nhưng `dp`
  hoàn toàn không phụ thuộc `dec` nên giả thuyết đó vô lý — kiểm tra lại
  bằng cách tái tạo đúng cơ chế `random_split` gốc (trước khi có
  `canonical_split.json`) cho MRF, thấy **tập train của MRF khác 570/12500
  câu (4.6%) so với tập baseline/SEQ đang dùng** (MRF train trước khi
  `canonical_split.json` tồn tại). Baseline và SEQ dùng đúng cùng 1 tập —
  và cả 2 đều từng cần clip; MRF dùng tập khác — chưa từng cần. Nhiều khả
  năng có 1-vài câu trong 570 câu khác biệt đó gây alignment/MAS cực đoan
  (phoneme bị gán độ dài gần-0), dẫn tới spline-bin gần-0 dễ triệt tiêu số
  học dưới bf16 hơn — không phải do kiến trúc resblock.

Baseline và MRF(1,1,1) **không cần train lại** — cả hai chưa từng chạm phải
lỗ hổng này trong suốt 1500 epoch (MRF do train trên tập dữ liệu khác;
baseline tự nó đã có clip từ epoch 89) nên kết quả hiện có không bị ảnh
hưởng.

**Áp dụng bản vá tail_bound (coqui-ai/TTS#1959) có cần train lại baseline
không?** — Không. Checkpoint cuối của baseline (epoch=1489, dùng cho mọi
so sánh/report) là **kết quả thật đã xảy ra trong quá khứ** — nếu quá
trình train ra nó từng chạm đúng edge-case "toàn bộ input ngoài
tail_bound", nó đã sập NaN ngay lúc đó, không thể có checkpoint khỏe mạnh
như hiện tại. Guard thêm sau này không thay đổi được kết quả một quá
trình đã xảy ra và đã hoàn tất — chỉ có ý nghĩa PHÒNG NGỪA cho các lần
train MỚI (như SEQ đang làm), không "sửa lại" được lịch sử baseline. Nếu
train baseline lại từ đầu, guard này nhiều khả năng giúp nó không cần các
lần resume-từ-checkpoint-sạch thủ công đã từng xảy ra (version_0→version_1
→version_2) — nhưng không có lý do để làm vậy khi baseline đã hoàn tất và
mọi eval (WER/UTMOSv2/RTF) đều dựa trên checkpoint đã có.

---

## 2. MOS đánh giá bởi người thật

UTMOSv2 hiện tại chỉ là MOS-predictor tự động. Cần vòng nghe thật (15-30
người, qua crowdsourcing) để claim chất lượng đứng vững trước phản biện
paper. Nằm ngoài khả năng tự thực hiện — cần user tổ chức riêng.

**Trạng thái**: chưa bắt đầu, cần user hành động.

---

## 3. So sánh với SOTA công bố ngoài project (tham khảo)

Đã có tham chiếu Vocos nội bộ (`paper_ready_report.md` §2.5) nhưng đo
trước khi sửa lỗi phương pháp luận — không phải so sánh chính thức.

**Trạng thái**: đã có tham chiếu sơ bộ, chưa làm chính thức.
