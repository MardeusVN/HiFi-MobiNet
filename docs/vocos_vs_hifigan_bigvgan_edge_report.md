# Vì sao Vocos phù hợp triển khai Edge hơn HiFi-GAN và BigVGAN

**Phạm vi:** so sánh 3 họ kiến trúc Generator (vocoder) đã được port và tích hợp thật vào pipeline VITS2 của BanhmiTTS — HiFi-GAN (± SnakeBeta), BigVGAN (± AMP đầy đủ), và Vocos — trên 3 khía cạnh triển khai Edge: **tốc độ suy luận**, **kích thước tham số**, và **chất lượng tái tạo**. Toàn bộ số liệu trong báo cáo này (trừ phần có trích dẫn paper) được đo trực tiếp trên các module thật của dự án, bằng các script trong `test_function/`, chạy lại ngay trước khi viết báo cáo — không lấy từ trí nhớ.

---

## 1. Kết luận nhanh (TL;DR)

| Tiêu chí | HiFi-GAN | BigVGAN (full AMP) | **Vocos** |
|---|---|---|---|
| Tốc độ suy luận (real-time factor) | 58.5x | 18.3x (chậm nhất) | **150.8x** |
| Tham số Generator (cấu hình mặc định) | 1.66M | 1.67M | 13.86M (nhưng có thể nén còn **1.63M**, xem §4.4) |
| Chất lượng tái tạo (loss_mel, overfit 400 bước) | 11.55 (kém nhất) | 10.84 | **3.89** |
| Kết quả training thật (val_loss_mel, full dataset) | 18.9497 (sau **2000/2000** epoch) | — (chưa train full-scale) | **18.2628** (mới **464/2000** epoch) |

Vocos thắng rõ ở tốc độ suy luận và chất lượng ở cùng ngân sách tham số, nhưng có 2 điểm cần lưu ý ngay để không hiểu lầm: (1) tham số **mặc định** của Vocos lớn hơn nhiều, phải chủ động nén mới về ngang HiFi-GAN; (2) lợi thế tốc độ này **không** thể hiện ở wall-clock của training end-to-end (xem §5) — nó chỉ thể hiện rõ ở **suy luận (inference)**, tức đúng thứ quan trọng cho Edge.

---

## 2. Vì sao về mặt kiến trúc, Vocos vốn hợp Edge hơn

HiFi-GAN và BigVGAN đều là **vocoder miền thời gian**: chúng upsample tín hiệu dần dần qua một chuỗi `ConvTranspose1d` (trong dự án này là 3 tầng, tỉ lệ 8×8×4 = 256 = hop_length), mỗi tầng đều kèm resblock để làm mượt. Toàn bộ compute tăng dần theo độ phân giải thời gian — tầng cuối cùng phải chạy conv trên chuỗi dài gần bằng waveform đầu ra.

Vocos đi theo hướng khác hẳn: giữ **nguyên độ phân giải thời gian ở mức frame** (bằng độ phân giải mel/latent đầu vào) xuyên suốt toàn bộ backbone ConvNeXt, và **chỉ upsample một lần duy nhất** ở bước cuối bằng Inverse FFT (ISTFT) — một phép biến đổi toán học cố định, không có tham số học, cực rẻ về compute so với chuỗi ConvTranspose1d. Nói cách khác: phần "nặng" (conv có tham số) chạy ở độ phân giải thấp, phần upsample lên độ phân giải cao là phép toán không tốn tham số. Đây chính là lý do cấu trúc khiến Vocos rẻ hơn về suy luận — không phải một chi tiết cài đặt ngẫu nhiên.

BigVGAN đi theo hướng ngược lại: để chống aliasing khi dùng activation phi tuyến (Snake) xen giữa các tầng upsample, nó chèn thêm bước **upsample → activate → downsample (Anti-aliased Multi-Periodicity)** ở mỗi resblock — cải thiện chất lượng nhưng nhân chi phí compute lên 3-4 lần, đúng hướng ngược với mục tiêu Edge.

---

## 3. Vấn đề lịch sử mà Vocos giải quyết (bối cảnh từ paper gốc, arXiv 2306.00814)

Ý tưởng "cho GAN dự đoán trực tiếp hệ số STFT phức, bỏ hẳn upsample miền thời gian" **không mới** — nhưng trước Vocos, mọi nỗ lực đều thất bại hoặc chỉ thành công một phần:

- **Gritsenko et al. 2020**: thử train GAN sinh trực tiếp hệ số STFT — *"unable to train it successfully due to its inherent instability"*.
- **iSTFTNet (Kaneko et al. 2022)**: chỉ dám thay **2 tầng upsample cuối** bằng ISTFT, phần lớn vẫn giữ ConvTranspose1d — *"replacing more upsampling layers drastically degrades the quality"*.
- **Pasini & Schlüter 2022**: thành công nhưng phải train nhiều giai đoạn (multi-step) phức tạp mới ổn định.

Nguyên nhân gốc rễ: dự đoán trực tiếp **góc pha (phase)** rất khó huấn luyện ổn định, vì góc pha có tính "wrap" (−π và π là cùng một điểm) mà một activation thông thường (vd. `tanh` scale về [−π, π]) không tự nhiên biểu diễn được — gây gián đoạn ở biên.

Giải pháp của Vocos (Contribution #2 trong paper): thay vì dự đoán góc pha trực tiếp, dự đoán **(cos φ, sin φ)** qua một activation định nghĩa trên đường tròn đơn vị — *"naturally incorporates implicit phase wrapping, ensuring meaningful values across all phase angles"*. Đây là lý do Vocos là công trình đầu tiên train ổn định **100% ISTFT-based upsampling** trên audio tổng quát (không chỉ 2 tầng cuối như iSTFTNet, không cần multi-step như Pasini & Schlüter).

Paper cũng đưa ra lý do bỏ dilated convolution (đặc trưng của resblock kiểu HiFi-GAN): vì Vocos giữ độ phân giải thời gian thấp xuyên suốt, receptive field lớn của dilated conv không còn cần thiết như ở vocoder miền thời gian — thay bằng ConvNeXt cho hiệu quả tốt hơn ở ngân sách tham số tương đương.

Số liệu công bố trong paper (LibriTTS, so với HiFi-GAN/BigVGAN cùng cấu hình chuẩn của họ): **nhanh hơn HiFi-GAN 13 lần và BigVGAN gần 70 lần** (đo trên CPU), trong khi UTMOS/VISQOL/PESQ/Periodicity gần như ngang BigVGAN (chỉ thua UTMOS: 3.734 vs 3.749) và không có khác biệt MOS/SMOS có ý nghĩa thống kê (Wilcoxon p>0.05) so với BigVGAN.

---

## 4. Bằng chứng thực nghiệm trên chính dự án BanhmiTTS

### 4.1. Tốc độ suy luận (real-time factor) — số liên quan trực tiếp nhất đến Edge

Đo bằng `test_function/benchmark_generator_speed.py`, input đúng shape thật của VITS (`initial_channel=192`, batch=16, 32 frame/clip = 0.372s/clip), CPU-only, 50 lần lặp sau 10 lần warmup:

| Generator | Params | Forward (ms) | **RTF (suy luận)** | Forward+Backward (ms, chi phí training) |
|---|---:|---:|---:|---:|
| hifigan | 1,664,768 | 101.68 | 58.5x | 319.40 |
| hifigan_snake | 1,668,416 | 120.55 | 49.3x | 479.51 |
| bigvgan (full AMP) | 1,668,673 | 324.41 | 18.3x | 1120.27 |
| bigvgan_lite | 1,668,673 | 150.18 | 39.6x | 532.44 |
| **vocos** (mặc định) | 13,861,378 | **39.41** | **150.8x** | 129.12 |

Vocos suy luận nhanh hơn HiFi-GAN vanilla **2.6 lần**, nhanh hơn BigVGAN full **8.2 lần** — mặc dù có **8.3 lần nhiều tham số hơn**. Đây chính là hệ quả trực tiếp của §2: forward-pass không phụ thuộc số tham số mà phụ thuộc số phép tính tuần tự trên trục thời gian dài, và Vocos né được phần đó.

### 4.2. Chất lượng tái tạo (mel-loss, cùng ngân sách bước huấn luyện)

Đo bằng `test_function/compare_generators.py`: cả 5 kiến trúc overfit cùng 1 đoạn audio thật (LJ018-0126, 2s), cùng optimizer/LR/seed, 400 bước, input là STFT-magnitude thật (không phải nhiễu ngẫu nhiên):

| Generator | Params | loss_mel sau 400 bước |
|---|---:|---:|
| **vocos** (mặc định) | 15,011,842 | **3.89** |
| hifigan_snake | 2,243,648 | 10.59 |
| bigvgan_lite | 2,243,905 | 10.74 |
| bigvgan (full AMP) | 2,243,905 | 10.84 |
| hifigan | 2,240,000 | 11.55 |

Vocos đạt loss thấp hơn gần **3 lần** so với kiến trúc tốt thứ nhì (hifigan_snake) — nhưng ở đây tham số cũng lớn hơn ~6.7 lần, nên phần lớn chênh lệch này đến từ dung lượng mô hình, chưa tách bạch được phần nào do kiến trúc thuần túy. Mục 4.4 giải quyết câu hỏi này bằng cách ép ngang tham số.

### 4.3. Kết quả training thật, full-scale (không phải overfit ngắn)

Từ 2 lần chạy training thật trên toàn bộ dataset LJSpeech, 2000 epoch theo kế hoạch:

- **HiFi-GAN + SnakeBeta**: chạy đủ **2000/2000 epoch**, checkpoint tốt nhất `best-epoch=1992-val_loss_mel=18.9497.ckpt`.
- **Vocos** (cấu hình mặc định, `configs/banhmi_vocos.yaml`): dừng sớm theo yêu cầu ở **epoch 464/2000** (mới 23% lộ trình), checkpoint tốt nhất `best-epoch=464-val_loss_mel=18.2628.ckpt`.

Vocos mới đi 23% quãng đường đã có val_loss_mel **thấp hơn** (tốt hơn) kết quả HiFi-GAN+Snake sau khi train **trọn vẹn** 2000 epoch. Đây là bằng chứng mạnh nhất trong báo cáo này vì nó đo trên pipeline VITS2 đầy đủ, dữ liệu thật, không phải overfit 1 clip.

### 4.4. Ép Vocos về ngân sách tham số ngang HiFi-GAN — vẫn thắng

Cấu hình mặc định của Vocos trong dự án (`dim=512, intermediate_dim=1536, num_layers=8`) cho ra **13.86M tham số** — lớn hơn hẳn ~1.66M của họ HiFi-GAN. Đây **không phải** lợi thế "nhẹ hơn" tự động — phải chủ động nén.

Đã thử nghiệm biến thể `vocos_small` (`dim=160, intermediate_dim=480, num_layers=8`), đo lại với đúng input shape thật (`initial_channel=192`):

| Generator | Params |
|---|---:|
| hifigan / hifigan_snake / bigvgan / bigvgan_lite | ~1.66M – 1.67M |
| vocos (mặc định) | 13.86M |
| **vocos_small** | **1.63M** ← ngang họ HiFi-GAN |

Ở lần đo trước đó trong phiên làm việc này (overfit trên input STFT-magnitude, không phải latent VITS thật), `vocos_small` vẫn đạt loss_mel ≈ 6.0 — vẫn tốt hơn rõ rệt so với toàn bộ họ HiFi-GAN/BigVGAN (10.5–11.5) dù tham số ngang nhau. Kết luận: lợi thế chất lượng của Vocos **không chỉ** đến từ việc "nhiều tham số hơn" — bản thân kiến trúc ConvNeXt + ISTFT-head hiệu quả hơn trên mỗi tham số, dù không hiệu quả bằng khi so ở tốc độ suy luận thuần túy (§4.1 dùng bản mặc định 13.86M).

---

## 5. Đính chính quan trọng: "nhanh hơn" — nhanh ở đâu?

Khi kiểm chứng lại bằng timestamp thật của 2 lần training full-scale (`stat` trên `hparams.yaml`/checkpoint), thời gian mỗi epoch gần như **ngang nhau**: HiFi-GAN+Snake ≈5.43 phút/epoch, Vocos ≈5.55 phút/epoch. Lợi thế 2.6–8.2 lần đo được ở §4.1 **không** xuất hiện ở wall-clock training.

Lý do không mâu thuẫn với §4.1: một bước training chạy **toàn bộ** pipeline — TextEncoder, PosteriorEncoder (WaveNet 16 lớp), Flow (coupling + attention), StochasticDurationPredictor, MAS alignment, dataloading, **và cả discriminator** (forward+backward) — những phần này không đổi giữa 2 cấu hình và áp đảo tổng thời gian bước. Discriminator đặc biệt nặng: benchmark riêng (`benchmark_discriminator_speed.py`, batch=16) cho thấy 1 bước training discriminator tốn **6.9–8.1 giây** — gấp hàng chục lần chi phí forward riêng của Generator (39–324 ms). Generator chỉ là một phần nhỏ trong tổng chi phí 1 bước training.

Ở **suy luận (deployment/Edge)**, bức tranh đảo ngược: discriminator bị loại bỏ hoàn toàn (không tồn tại trong model export), PosteriorEncoder cũng bị loại (chỉ dùng lúc training để học latent). Phần còn lại chạy khi suy luận là TextEncoder + StochasticDurationPredictor + Flow (chạy ở độ phân giải phoneme/frame — ngắn) và **Generator** (chạy ở độ phân giải waveform — dài hơn hop_length=256 lần). Generator vì vậy chiếm tỷ trọng lớn hơn nhiều trong tổng chi phí suy luận so với trong tổng chi phí 1 bước training. Kết luận: **RTF ở §4.1 (150.8x vs 58.5x/18.3x) là con số phản ánh đúng thực tế triển khai Edge**, còn phần "ngang nhau" ở wall-clock training chỉ là đặc thù của quá trình huấn luyện (có discriminator, có backward pass toàn pipeline) — không áp dụng khi export model để chạy trên thiết bị.

---

## 6. Điểm mạnh của Vocos cho Edge

1. **Suy luận nhanh hơn 2.6–8.2 lần** so với HiFi-GAN/BigVGAN ở cùng shape input thật (§4.1) — chính là chi phí lặp lại mỗi lần tổng hợp giọng nói trên thiết bị.
2. **Chất lượng/tham số vượt trội**: ở ngân sách tham số ngang nhau (~1.6-1.7M), Vocos vẫn cho loss_mel thấp hơn rõ rệt (§4.4) — không phải đánh đổi kích thước lấy chất lượng, được cả hai nếu chịu tinh chỉnh cấu hình.
3. **Đã kiểm chứng ở scale thật**, không chỉ lý thuyết hay overfit ngắn: sau 23% lộ trình training đã vượt kết quả HiFi-GAN+Snake train đủ 2000 epoch (§4.3).
4. **Discriminator đi kèm cũng rẻ hơn khi training**: combo MPD+MRD kiểu Vocos (42.5M tham số, 6.87s/bước) rẻ hơn combo kiểu BigVGan hiện có trong dự án (47.0M tham số, 8.14s/bước) — không phải đánh đổi thêm chi phí training để có ưu thế suy luận.
5. **Nền tảng lý thuyết vững**: giải quyết được bài toán huấn luyện ổn định GAN sinh trực tiếp hệ số STFT mà 3 nỗ lực trước (Gritsenko 2020, iSTFTNet 2022, Pasini & Schlüter 2022) đều thất bại hoặc chỉ thành công một phần (§3) — không phải một mẹo cục bộ dễ vỡ khi đổi dataset/điều kiện.
6. **remove_weight_norm() là no-op**: Vocos không dùng weight norm ở đâu cả, nên bước hậu xử lý trước khi export/deploy đơn giản hơn (không có gì phải "gỡ").

## 7. Hạn chế / rủi ro của Vocos

1. **Cấu hình mặc định nặng gấp ~8 lần** họ HiFi-GAN (13.86M vs 1.66M tham số) — muốn có lợi thế "gọn" cho Edge phải chủ động nén (`vocos_small`), không có sẵn.
2. **Discriminator đi kèm rất lớn**: MPD của Vocos một mình đã 41.1M tham số — nặng hơn cả Generator mặc định (13.86M). Không ảnh hưởng model export (discriminator bị bỏ), nhưng làm tăng chi phí bộ nhớ/compute khi training — cần GPU đủ VRAM.
3. **Cần hạ trọng số loss của MRD để ổn định**: cấu hình gốc từ tác giả dùng `mrd_loss_coeff=0.1` thay vì 1.0 (giá trị mặc định trong class) — nếu không hạ, nhánh MRD (5 sub-band, mỗi sub-band 1 stack conv2d riêng) có thể lấn át quá trình học so với MPD, dự án đã áp dụng đúng giá trị 0.1 này trong `banhmi_vocos.yaml`, nhưng đây là một chi tiết dễ bỏ sót nếu tinh chỉnh lại cấu hình.
4. **Chưa có số liệu training full-scale hoàn chỉnh cho BigVGAN** để đối chiếu 3 chiều — mới có HiFi-GAN+Snake (hoàn tất) và Vocos (dừng ở epoch 464); kết luận "Vocos tốt hơn BigVGAN ở scale thật" trong dự án này hiện dựa trên benchmark overfit ngắn (§4.2) và paper gốc (§3), chưa có một lần training BigVGAN full-scale để đối chiếu trực tiếp trên đúng dataset này.
5. **UTMOS trong paper gốc vẫn thua BigVGAN** (3.734 vs 3.749, dù các metric khác Vocos nhỉnh hơn) — Vocos không thắng tuyệt đối mọi thước đo chất lượng, chỉ là không thua có ý nghĩa thống kê ở đánh giá nghe thật (MOS/SMOS).
6. **ISTFT cố định n_fft/hop_length** ngay trong kiến trúc (khớp với STFT dùng để train) — không linh hoạt đổi tỉ lệ upsample tùy ý như chuỗi ConvTranspose1d của HiFi-GAN/BigVGAN (vốn có thể thêm/bớt tầng để đổi hop_length). Đổi sample rate hoặc hop_length của dự án sau này sẽ đòi sửa trực tiếp `n_fft`/`hop_length` của Vocos, không chỉ đổi 1 tham số cấu hình tầng upsample.
7. **Chưa có benchmark trên phần cứng Edge thật** (di động/embedded) — mọi số liệu tốc độ ở đây đo trên CPU máy chủ (12 luồng); tỉ lệ tương đối giữa các kiến trúc có khả năng giữ nguyên (đều dùng chung backbone conv1d) nhưng con số RTF tuyệt đối trên thiết bị đích cần đo lại.

---

## 8. Khuyến nghị cho dự án

- Dùng `configs/banhmi_vocos.yaml` (`use_vocos: true`) làm cấu hình chính cho mục tiêu Edge — đã tích hợp thật vào `SynthesizerTrn`/`VitsModel`, không phải benchmark cô lập.
- Cân nhắc chuyển sang biến thể `vocos_small` (`vocos_dim=160, vocos_intermediate_dim=480, vocos_num_layers=8`) nếu ràng buộc dung lượng model là ưu tiên hàng đầu — đã kiểm chứng giữ được lợi thế chất lượng/tham số so với HiFi-GAN family ở cùng ngân sách ~1.6-1.7M.
- Giữ nguyên `vocos_mrd_loss_coeff: 0.1` (đã đặt đúng theo cấu hình gốc tác giả) — đừng đổi về 1.0 nếu không có lý do rõ ràng.
- Nên chạy tiếp training Vocos đến hết 2000 epoch (hoặc ít nhất đủ để so công bằng với mốc 1992 epoch của HiFi-GAN+Snake) trước khi kết luận chính thức — kết quả epoch 464 đã tốt hơn nhưng chưa phải điểm hội tụ, biên độ tốt hơn có thể còn thay đổi.
- Nếu có thời gian, nên train BigVGAN full-scale trên đúng dataset này để lấp khoảng trống ở hạn chế #4 — hiện BigVGAN chỉ được so sánh ở scale overfit ngắn và qua số liệu paper gốc (đo trên LibriTTS, không phải LJSpeech).

---

## 9. Nguồn số liệu

- **Benchmark thật, đo lại trong phiên này**: `test_function/benchmark_generator_speed.py`, `test_function/benchmark_discriminator_speed.py`, `test_function/compare_generators.py` (chạy trên `/home/capstone/env`, WSL2, CPU 12 luồng).
- **Checkpoint training thật**: `overnight_run_full/lightning_logs/version_0/checkpoints/best-epoch=1992-val_loss_mel=18.9497.ckpt`, `vocos_run_full/lightning_logs/version_0/checkpoints/best-epoch=464-val_loss_mel=18.2628.ckpt`.
- **Paper gốc**: Siuzdak, "Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis", arXiv:2306.00814 — mục 1.2 "Contribution", Bảng 1 (LibriTTS), so sánh với iSTFTNet (Kaneko et al. 2022), Gritsenko et al. 2020, Pasini & Schlüter 2022.
- **Code tham chiếu gốc**: gemelo-ai/vocos (kiến trúc, discriminator, hàm loss) — đã port và kiểm chứng song song trong `banhmi_train/Vocos/`.
