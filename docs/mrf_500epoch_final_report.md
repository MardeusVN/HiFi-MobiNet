# Report: Kết quả training Banhmi-TTS (500 epochs) so với Piper (1500 epochs)

## 1. Kết luận nhanh

| | HiFi-GAN baseline | MRF (1,1,1) | Chênh lệch |
|---|---:|---:|---|
| Generator params | 1.664.768 | **1.157.024** | nhẹ hơn 30.5% |
| Generator size (fp32) | 6.35 MB | **4.41 MB** | nhẹ hơn 1.94 MB |
| WER ↓ | 0.1206 | **0.1110** | tốt hơn ~8.0% |
| UTMOSv2 ↑ | 3.136 | **3.348** | tốt hơn ~6.8% |
| RTF ↓ (CPU, đo sạch) | 0.0429 | **0.0373** | nhanh hơn ~13.0% |

MRF(1,1,1) thắng cả 4 trục: nhẹ hơn, nhanh hơn, chính xác hơn (WER thấp hơn), và điểm chất lượng cảm nhận cao hơn (UTMOSv2) dù mới train 500 epochs so với baseline đã hội tụ 1500 epochs. Cả hai số RTF đều đo trong điều kiện CPU free.

## 2. Kiến trúc thay đổi gì

**Scaffold Generator** (conv_pre, 3 tầng ConvTranspose1d upsample rate **8×8×4**, conv_post) giữ nguyên 100% giữa 2 model:

- Cùng upsample_initial_channel = 256
- Cùng upsample_kernel_sizes = [16, 16, 8]
- Cùng resblock_kernel_sizes = [3, 5, 7] (dùng làm 3 kernel nhánh MRF ở cả 2 phía)

**Khác biệt duy nhất** là cơ chế resblock áp dụng sau mỗi tầng upsample:

- Baseline (resblock="2"): mỗi tầng có 3 nhánh ResBlock2 (mỗi nhánh 2 cặp dilated-Conv1d dense, kernel 3/5/7 tương ứng dilation [1,2]/[2,6]/[3,12]), chạy song song, cộng rồi chia 3. Activation LeakyReLU.
- MRF (resblock="mrf"): mỗi tầng có 3 nhánh ResBlockInverted (Pointwise-expand → weight_norm → SnakeBeta → Depthwise(kernel 3/5/7 tương ứng) → weight_norm → SnakeBeta → Pointwise-project linear, zero-init → weight_norm, + residual), expansion=1 ở mọi nhánh/tầng (không giãn kênh), cùng chạy song song cộng chia 3. Activation SnakeBeta xuyên suốt.

## 3. Lý do dùng weight_norm thay vì BatchNorm cho ResBlockInverted

**1. Train/inference mismatch**

GAN vốn đã bất ổn, BatchNorm làm trầm trọng thêm. BatchNorm hành xử khác nhau giữa train (dùng thống kê batch hiện tại) và eval (dùng running mean/var tích luỹ, đóng băng sau train). Running stats được tích luỹ trong lúc GAN training dao động mạnh (chính ta vừa quan sát loss_gen_all dao động 37-43 trong log thật) có thể là ước lượng kém cho phân phối thật ở inference — đây là lý do đã được ghi nhận rộng rãi trong literature GAN (từ sau DCGAN dùng BatchNorm bất ổn), khiến hầu hết vocoder GAN hiện đại (HiFi-GAN, WaveGAN, BigVGAN) bỏ hẳn BatchNorm trong generator.

**2. weight_norm loại bỏ được hoàn toàn ở inference còn BatchNorm thì không**

remove_weight_norm() (đã gọi trước mọi lần infer() trong toàn bộ session này) gấp weight = weight_g * weight_v/||weight_v|| thành một tensor weight phẳng duy nhất, không tốn thêm phép tính hay tham số nào lúc suy luận. BatchNorm ở eval mode vẫn phải lưu và áp dụng running_mean/running_var/weight/bias (4 tensor/kênh), tức là thêm tham số + thêm phép affine mỗi kênh mỗi lần forward, đi ngược mục tiêu chính của cả hướng novelty này (tối thiểu tham số và tối thiểu compute lúc suy luận).

**3. Cơ chế zero-init "identity no-op lúc khởi tạo" chỉ hoạt động sạch với weight_norm**

Root-cause fix quan trọng nhất của ResBlockInverted (đã phát hiện giữa session) là zero-init pw_project.weight_g để cả khối bắt đầu như một residual no-op, không phá vỡ phần mạng đã hội tụ phía trước, cùng convention "ghép vào không xáo trộn" cho phép việc này chính xác vì weight_g=0 khiến weight hiệu dụng bằng 0 tuyệt đối, bất kể weight_v. BatchNorm không có cơ chế tương đương sạch như vậy, dù có zero-init affine gamma/beta, bước chuẩn hoá theo thống kê batch (trừ mean, chia std) vẫn chủ động biến đổi activation dựa trên dữ liệu batch, tức vẫn là một "nhiễu loạn ngẫu nhiên", không phải identity thật.

**4. BatchNorm phức tạp hoá DDP** (2 GPU dùng xuyên suốt project). BatchNorm chuẩn trên nhiều GPU cho ra thống kê cục bộ khác nhau mỗi replica (không đồng nhất), muốn đúng phải dùng SyncBatchNorm (thêm all-reduce mỗi forward pass, thêm một cấu hình cần bật/tắt, thêm điểm có thể lệch giữa chạy 1-GPU và multi-GPU). weight_norm không có khái niệm "thống kê theo batch" nên hoàn toàn miễn nhiễm vấn đề này, không cần đồng bộ gì giữa các GPU.

**Tóm lại:** weight_norm phù hợp hơn ở mọi khía cạnh liên quan trực tiếp đến bối cảnh dự án này — ổn định GAN ở batch nhỏ, nhất quán với toàn bộ phần còn lại của Generator, miễn phí ở inference, tương thích với cơ chế zero-init đã xác lập, và không phức tạp hoá DDP — không phải một lựa chọn tuỳ tiện mà là hệ quả trực tiếp của các ràng buộc đã có sẵn trong codebase.

## 4. Vì sao chọn expansion=(1,1,1) cho MRF (không phải (2,2,1)/(4,4,1)/(6,6,1))

### 4.1. Vì sao thiết kế "giảm dần theo tầng" (kiểu 6,6,1) chứ không đồng nhất (6,6,6)

**Nguyên tắc gốc:** chi phí của mỗi nhánh ResBlockInverted (pointwise-expand + depthwise + pointwise-project) tỉ lệ với expansion × T, với T là số frame tại tầng đó. Ba tầng upsample có T tăng dần rất nhanh (256 → 2048 → 8192, nhân 8×8×4), trong khi số kênh giảm dần (256→128→64→32) — nghĩa là tầng cuối cùng (T=8192, kênh chỉ còn 32) là nơi một đơn vị expansion tốn kém nhất theo thời gian thực, dù kênh ở đó nhỏ nhất.

Với MRF, vấn đề này còn bị khuếch đại thêm một lớp nữa: mỗi tầng không chỉ chạy 1-2 khối nối tiếp mà chạy 3 nhánh song song cùng lúc (kernel 3, 5, 7), nên chi phí pointwise/depthwise ở tầng cuối bị nhân thêm hệ số 3 so với 1 khối đơn. Giữ expansion cao đồng nhất ở mọi tầng (kiểu 6,6,6) sẽ khiến đúng tầng đắt nhất (tầng cuối) phải trả chi phí expansion=6 × 3 nhánh × T=8192, đắt nhất trong toàn bộ generator. Vì vậy nguyên tắc "giảm expansion về 1 chỉ ở tầng cuối, giữ cao ở 2 tầng đầu (T nhỏ, giãn kênh rẻ)" dạng (X,X,1) được ưu tiên thử trước tiên thay vì đồng nhất (X,X,X), và tất cả các schedule MRF thử nghiệm trong báo cáo này đều theo dạng đó.

### 4.2. Kết quả đo 4 schedule (X,X,1) đã thử

| Expansion | Params | loss_mel (Final, overfit 800 bước) | Tốc độ 1 luồng | Tốc độ 12 luồng | So với baseline (24.29ms / 19.94ms) |
|---|---:|---:|---:|---:|---|
| (6,6,1) | 2.384.096 | 9.3273 | 41.40ms | 20.06ms | ~hòa (12 luồng), chậm hơn nhiều (1 luồng) |
| (4,4,1) | 2.123.360 | **8.9734** (tốt nhất) | 30.58ms | 19.53ms | ~hòa, không có lợi thế tốc độ rõ |
| (2,2,1) | 1.862.624 | 9.4520 (tệ nhất) | 16.23ms | 20.84ms | chậm hơn ở 12 luồng (đảo chiều) |
| **(1,1,1)** | **1.732.256** | 9.0788 | **14.00ms** | **17.30ms** | **nhanh hơn rõ ở cả 2 điều kiện** |

Dù xuất phát từ nguyên tắc "giảm dần về 1 ở tầng cuối", ngay cả (6,6,1) — schedule bảo thủ nhất trong 4 lựa chọn — vẫn không giữ được lợi thế tốc độ so với baseline khi nhân với 3 nhánh song song của MRF. (4,4,1) cho loss_mel tốt nhất nhưng cũng chỉ hòa vốn tốc độ. (2,2,1) tệ ở cả 2 trục. Chỉ (1,1,1) — bỏ hẳn cơ chế "expand" của MobileNetV2, gần với MobileNetV1/depthwise-separable thuần — mới giữ được lợi thế tốc độ nhất quán và rõ ràng ở cả 2 điều kiện đo luồng.

### 4.3. Chốt lại: vì sao (1,1,1)

**Ba lý do:**

- Duy nhất giữ được lợi thế tốc độ đáng tin cậy. Đây là tiêu chí quan trọng nhất vì mục tiêu gốc của cả hướng novelty này là tìm resblock vừa nhẹ vừa nhanh hơn baseline — (2,2,1)/(4,4,1)/(6,6,1) đều đánh mất mục tiêu này ở các mức độ khác nhau khi áp dụng cơ chế MRF song song thật.
- Loss_mel không tệ nhất, và khoảng cách với schedule tốt nhất (4,4,1) rất nhỏ (9.0788 vs 8.9734, ~1.2%) trong khi (4,4,1) tốn thêm ~22% tham số và không có lợi thế tốc độ — đánh đổi không xứng đáng.
- Nhẹ nhất trong 4 lựa chọn (1.732.256 tham số, ít hơn baseline 2.240.000 khoảng 22.7% ngay cả ở dạng harness test — còn nhẹ hơn nữa trong cấu hình production thật).

**Tóm lại:** Kết quả training 500 epoch thật sau đó xác nhận lựa chọn này đúng đắn: (1,1,1) không chỉ nhanh hơn mà còn thắng cả WER và UTMOSv2 so với baseline.

## 5. Tính toán chi tiết param & size

| Thành phần | Baseline (resblock="2") | MRF (resblock="mrf", expansion=1,1,1) |
|---|---:|---:|
| conv_pre (Conv1d 192→256, k=7) | 344.320 | 344.320 (giống hệt) |
| ups (3× ConvTranspose1d, upsample 8/8/4) | 672.416 | 672.416 (giống hệt) |
| pre_up_snakes (activation trước mỗi upsample) | 0 (LeakyReLU, không tham số) | 896 (SnakeBeta có α/β học được mỗi kênh) |
| **Resblock (phần thay đổi)** | **647.808** (9× ResBlock2 = 3 tầng × 3 nhánh dense dilated-conv) | **139.104** (9× ResBlockInverted = 3 tầng × 3 nhánh depthwise-separable) |
| ↳ tầng 0 (256→128 kênh, T nhỏ nhất) | (gộp trong tổng trên) | 104.064 |
| ↳ tầng 1 (128→64 kênh) | (gộp trong tổng trên) | 27.456 |
| ↳ tầng 2 (64→32 kênh, T lớn nhất) | (gộp trong tổng trên) | 7.584 |
| final_snake | 0 (LeakyReLU) | 64 (SnakeBeta) |
| conv_post (Conv1d →1, k=7, không bias) | 224 | 224 (giống hệt) |
| **TỔNG** | **1.664.768** | **1.157.024** |

**Chênh lệch: -507.744 tham số (-30.50%)**, toàn bộ chênh lệch nằm ở khối resblock (647.808 → 139.104, **giảm 78.5%**), vì:

- ResBlock2 dùng conv dense (mọi kênh nối mọi kênh) ở cả 2 conv/nhánh.
- ResBlockInverted ở expansion=1 dùng depthwise conv (1 tham số/kênh/vị trí kernel, không nối chéo kênh) làm phần tốn kernel-size nhất, chỉ 2 pointwise conv (1×1, dense nhưng không nhân kernel_size) bao quanh.
- Số tham số resblock giảm dần rất nhanh theo tầng (104K → 27K → 7.6K) vì kênh giảm theo cấp số nhân (256→128→64→32) trong khi depthwise scale tuyến tính theo kênh (không phải bình phương như dense conv). Đây chính là core lý do MobileNetV2-style depthwise-separable rẻ hơn dense đáng kể ở cùng độ rộng kênh.

## 6. Kết quả eval 500-mẫu cuối cùng (đã đo sạch)

| | HiFi-GAN baseline | MRF (1,1,1) |
|---|---:|---:|
| WER (mean) | 0.1206 | **0.1110** |
| UTMOSv2 (mean) | 3.136 | **3.348** |
| RTF (mean, CPU) | 0.0429 | **0.0373** |

## 7. Kết luận

MobileNetV2-style ResBlockInverted dùng làm nhánh cho cơ chế MRF thật của HiFi-GAN (expansion=1, kernel song song 3/5/7) là một thay thế thắng cả 4 trục so với resblock gốc: nhẹ hơn 30.5% tham số, nhanh hơn 13% (đo sạch), WER thấp hơn 8%, UTMOSv2 cao hơn 6.8% dù mới train 480/500 epoch so với baseline 1489/1500. Training hiện đã được resume tới 1000 epoch để xem liệu khoảng cách val_loss_mel còn lại (20.22 vs 19.79) có tiếp tục thu hẹp, và liệu lợi thế WER/UTMOSv2 hiện tại có được giữ vững hay mở rộng thêm khi training lâu hơn.
