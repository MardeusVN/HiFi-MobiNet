"""PyTorch Lightning training loop for the single-speaker VITS model.

Single-speaker is not a runtime option here: banhmi_train.preprocess's
config.json has no `num_speakers`/`speaker_id_map` fields at all (removed
since this project permanently targets single-speaker LJSpeech), so
n_speakers=1 / gin_channels=0 are hardcoded rather than read from anywhere
-- there is no other value they could ever take.
"""
import argparse
import itertools
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from torch import autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from ..mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from .dataset import Batch, UtteranceCollate, VitsDataset
from .losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from .modules.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from .modules.duration_discriminator import DurationDiscriminator
from .modules.synthesizer import SynthesizerTrn
from .utils.commons import slice_segments

_LOGGER = logging.getLogger("banhmi_train.vits.training")

_N_SPEAKERS = 1
_GIN_CHANNELS = 0


class VitsModel(pl.LightningModule):
    def __init__(
        self,
        num_symbols: int,
        # audio
        resblock: str = "2",
        resblock_kernel_sizes: Tuple[int, ...] = (3, 5, 7),
        resblock_dilation_sizes: Tuple[Tuple[int, ...], ...] = ((1, 2), (2, 6), (3, 12)),
        upsample_rates: Tuple[int, ...] = (8, 8, 4),
        upsample_initial_channel: int = 256,
        upsample_kernel_sizes: Tuple[int, ...] = (16, 16, 8),
        # mel
        filter_length: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        mel_channels: int = 80,
        sample_rate: int = 22050,
        mel_fmin: float = 0.0,
        mel_fmax: Optional[float] = None,
        # model
        inter_channels: int = 192,
        hidden_channels: int = 192,
        filter_channels: int = 768,
        n_heads: int = 2,
        n_layers: int = 6,
        kernel_size: int = 3,
        p_dropout: float = 0.1,
        use_spectral_norm: bool = False,
        use_sdp: bool = True,
        segment_size: int = 8192,
        # PosteriorEncoder is training-only (discarded before ONNX export),
        # so its size only trades off training speed/memory against how
        # good a training signal it gives the flow -- never inference cost.
        posterior_encoder_kernel_size: int = 5,
        posterior_encoder_dilation_rate: int = 1,
        posterior_encoder_layers: int = 12,
        # Flow (ResidualCouplingBlock): part of the deployed/exported model,
        # so these trade off real inference speed/quality, unlike the
        # PosteriorEncoder knobs above.
        flow_kernel_size: int = 5,
        flow_dilation_rate: int = 1,
        flow_n_flows: int = 4,
        use_transformer_flows: bool = True,
        # VITS2's noise-scaled MAS: training-only (the alignment search
        # itself never runs at inference -- infer() uses the duration
        # predictor + generate_path instead), decays linearly to 0 over
        # `global_step`, so it only affects how training explores
        # alignments early on, not the final converged behavior.
        use_noise_scaled_mas: bool = False,
        mas_noise_scale_initial: float = 0.01,
        mas_noise_scale_decay: float = 2e-6,
        # Not part of EdgeTTS (which uses plain Snake1d) -- a BanhmiTTS-only
        # addition. Part of the exported model (real inference tradeoff).
        use_snake_beta: bool = True,
        # Discriminators are training-only (never exported), so these only
        # trade off training speed/memory against how much adversarial
        # signal the generator gets -- same category as PosteriorEncoder.
        use_mrd: bool = True,
        use_duration_discriminator: bool = True,
        # training
        dataset: Optional[List[Union[str, Path]]] = None,
        learning_rate: float = 2e-4,
        betas: Tuple[float, float] = (0.8, 0.99),
        eps: float = 1e-9,
        batch_size: int = 1,
        lr_decay: float = 0.999875,
        c_mel: int = 45,
        c_kl: float = 1.0,
        num_workers: int = 1,
        seed: int = 1234,
        # A real 3-way split: `test` is held out entirely until after
        # training finishes (see test_step/trainer.test), never touched for
        # monitoring or tuning decisions. Fixed counts, not fractions --
        # LJSpeech's size is permanently known (13,100 utterances), so a
        # hardcoded absolute split is as valid as a percentage and doesn't
        # silently balloon/shrink if the dataset ever changes size.
        # num_audio_samples is unrelated -- just a handful of *validation*
        # utterances synthesized and logged as listenable audio during
        # training, not a metric of any kind.
        num_val_examples: int = 500,
        num_test_examples: int = 500,
        num_audio_samples: int = 5,
        max_phoneme_ids: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model_g = SynthesizerTrn(
            n_vocab=self.hparams.num_symbols,
            spec_channels=self.hparams.filter_length // 2 + 1,
            segment_size=self.hparams.segment_size // self.hparams.hop_length,
            inter_channels=self.hparams.inter_channels,
            hidden_channels=self.hparams.hidden_channels,
            filter_channels=self.hparams.filter_channels,
            n_heads=self.hparams.n_heads,
            n_layers=self.hparams.n_layers,
            kernel_size=self.hparams.kernel_size,
            p_dropout=self.hparams.p_dropout,
            resblock=self.hparams.resblock,
            resblock_kernel_sizes=self.hparams.resblock_kernel_sizes,
            resblock_dilation_sizes=self.hparams.resblock_dilation_sizes,
            upsample_rates=self.hparams.upsample_rates,
            upsample_initial_channel=self.hparams.upsample_initial_channel,
            upsample_kernel_sizes=self.hparams.upsample_kernel_sizes,
            n_speakers=_N_SPEAKERS,
            gin_channels=_GIN_CHANNELS,
            use_sdp=self.hparams.use_sdp,
            posterior_encoder_kernel_size=self.hparams.posterior_encoder_kernel_size,
            posterior_encoder_dilation_rate=self.hparams.posterior_encoder_dilation_rate,
            posterior_encoder_layers=self.hparams.posterior_encoder_layers,
            flow_kernel_size=self.hparams.flow_kernel_size,
            flow_dilation_rate=self.hparams.flow_dilation_rate,
            flow_n_flows=self.hparams.flow_n_flows,
            use_transformer_flows=self.hparams.use_transformer_flows,
            use_snake_beta=self.hparams.use_snake_beta,
        )
        self.model_d = MultiPeriodDiscriminator(use_spectral_norm=self.hparams.use_spectral_norm)
        self.model_d_mrd = (
            MultiResolutionDiscriminator(use_spectral_norm=self.hparams.use_spectral_norm)
            if self.hparams.use_mrd
            else None
        )
        self.model_d_dur = (
            DurationDiscriminator(
                in_channels=self.hparams.hidden_channels,
                filter_channels=self.hparams.hidden_channels,
                kernel_size=3,
                p_dropout=self.hparams.p_dropout,
                gin_channels=_GIN_CHANNELS,
            )
            if self.hparams.use_duration_discriminator
            else None
        )

        self._train_dataset: Optional[Dataset] = None
        self._val_dataset: Optional[Dataset] = None
        self._test_dataset: Optional[Dataset] = None
        self._audio_sample_dataset: Optional[Dataset] = None
        self._load_datasets(num_val_examples, num_test_examples, num_audio_samples, max_phoneme_ids)

        # State kept between the generator and discriminator optimizer steps
        self._y = None
        self._y_hat = None
        self._dur_x = None
        self._dur_mask = None
        self._dur_real = None
        self._dur_fake = None

    def _load_datasets(
        self,
        num_val_examples: int,
        num_test_examples: int,
        num_audio_samples: int,
        max_phoneme_ids: Optional[int],
    ):
        if not self.hparams.dataset:
            _LOGGER.debug("No dataset to load")
            return

        full_dataset = VitsDataset(self.hparams.dataset[0], max_phoneme_ids=max_phoneme_ids)
        train_size = len(full_dataset) - num_val_examples - num_test_examples
        if train_size <= 0:
            raise ValueError(
                f"num_val_examples ({num_val_examples}) + num_test_examples ({num_test_examples}) "
                f">= dataset size ({len(full_dataset)}) -- nothing left to train on"
            )

        self._train_dataset, self._val_dataset, self._test_dataset = random_split(
            full_dataset, [train_size, num_val_examples, num_test_examples]
        )
        # A handful of *validation* utterances (never the held-out test set)
        # purely for logging listenable audio samples during training.
        num_audio_samples = min(num_audio_samples, len(self._val_dataset))
        self._audio_sample_dataset = Subset(self._val_dataset, range(num_audio_samples))

    def forward(self, text, text_lengths, scales):
        noise_scale, length_scale, noise_scale_w = scales[0], scales[1], scales[2]
        audio, *_ = self.model_g.infer(
            text, text_lengths, noise_scale=noise_scale, length_scale=length_scale, noise_scale_w=noise_scale_w
        )
        return audio

    def _dataloader(self, dataset):
        return DataLoader(
            dataset,
            collate_fn=UtteranceCollate(segment_size=self.hparams.segment_size),
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
        )

    def train_dataloader(self):
        return self._dataloader(self._train_dataset)

    def val_dataloader(self):
        return self._dataloader(self._val_dataset)

    def test_dataloader(self):
        return self._dataloader(self._test_dataset)

    def training_step(self, batch: Batch, batch_idx: int, optimizer_idx: int):
        if optimizer_idx == 0:
            return self.training_step_g(batch)
        return self.training_step_d(batch)

    def _current_mas_noise_scale(self) -> float:
        if not self.hparams.use_noise_scaled_mas:
            return 0.0
        scale = self.hparams.mas_noise_scale_initial - self.global_step * self.hparams.mas_noise_scale_decay
        return max(scale, 0.0)

    def training_step_g(self, batch: Batch):
        x, x_lengths = batch.phoneme_ids, batch.phoneme_lengths
        y, spec, spec_lengths = batch.audios, batch.spectrograms, batch.spectrogram_lengths

        mas_noise_scale = self._current_mas_noise_scale()
        (
            y_hat,
            l_length,
            _attn,
            ids_slice,
            x_mask,
            z_mask,
            (_z, z_p, m_p, logs_p, _m_q, logs_q),
            (x_hidden, logw, logw_),
        ) = self.model_g(x, x_lengths, spec, spec_lengths, mas_noise_scale=mas_noise_scale)
        self._y_hat = y_hat
        if self.hparams.use_noise_scaled_mas:
            self.log("mas_noise_scale", mas_noise_scale)

        # Saved for training_step_d's duration discriminator
        self._dur_x, self._dur_mask, self._dur_real, self._dur_fake = x_hidden, x_mask, logw_, logw

        mel = spec_to_mel_torch(
            spec, self.hparams.filter_length, self.hparams.mel_channels,
            self.hparams.sample_rate, self.hparams.mel_fmin, self.hparams.mel_fmax,
        )
        y_mel = slice_segments(mel, ids_slice, self.hparams.segment_size // self.hparams.hop_length)
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1), self.hparams.filter_length, self.hparams.mel_channels,
            self.hparams.sample_rate, self.hparams.hop_length, self.hparams.win_length,
            self.hparams.mel_fmin, self.hparams.mel_fmax,
        )
        y = slice_segments(y, ids_slice * self.hparams.hop_length, self.hparams.segment_size)
        self._y = y

        _y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.model_d(y, y_hat)

        with autocast(self.device.type, enabled=False):
            loss_dur = torch.sum(l_length.float())
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * self.hparams.c_mel
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * self.hparams.c_kl

            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, _ = generator_loss(y_d_hat_g)
            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl

            self.log("loss_mel", loss_mel)
            self.log("loss_kl", loss_kl)
            self.log("loss_dur", loss_dur)
            self.log("loss_gen", loss_gen)
            self.log("loss_fm", loss_fm)

            if self.model_d_mrd is not None:
                _y_d_hat_r_mrd, y_d_hat_g_mrd, fmap_r_mrd, fmap_g_mrd = self.model_d_mrd(y, y_hat)
                loss_fm_mrd = feature_loss(fmap_r_mrd, fmap_g_mrd)
                loss_gen_mrd, _ = generator_loss(y_d_hat_g_mrd)
                loss_gen_all = loss_gen_all + loss_gen_mrd + loss_fm_mrd
                self.log("loss_gen_mrd", loss_gen_mrd)

            if self.model_d_dur is not None:
                _dur_probs_r, dur_probs_hat = self.model_d_dur(x_hidden, x_mask, logw_, logw)
                loss_dur_gen, _ = generator_loss(dur_probs_hat)
                loss_gen_all = loss_gen_all + loss_dur_gen
                self.log("loss_dur_gen", loss_dur_gen)

            self.log("loss_gen_all", loss_gen_all)
            return loss_gen_all

    def training_step_d(self, batch: Batch):
        y, y_hat = self._y, self._y_hat
        y_d_hat_r, y_d_hat_g, _, _ = self.model_d(y, y_hat.detach())

        with autocast(self.device.type, enabled=False):
            loss_disc, *_ = discriminator_loss(y_d_hat_r, y_d_hat_g)
            loss_disc_all = loss_disc
            self.log("loss_disc", loss_disc)

            if self.model_d_mrd is not None:
                y_d_hat_r_mrd, y_d_hat_g_mrd, _, _ = self.model_d_mrd(y, y_hat.detach())
                loss_disc_mrd, *_ = discriminator_loss(y_d_hat_r_mrd, y_d_hat_g_mrd)
                loss_disc_all = loss_disc_all + loss_disc_mrd
                self.log("loss_disc_mrd", loss_disc_mrd)

            if self.model_d_dur is not None:
                dur_probs_r, dur_probs_hat = self.model_d_dur(
                    self._dur_x.detach(), self._dur_mask, self._dur_real.detach(), self._dur_fake.detach()
                )
                loss_disc_dur, *_ = discriminator_loss(dur_probs_r, dur_probs_hat)
                loss_disc_all = loss_disc_all + loss_disc_dur
                self.log("loss_disc_dur", loss_disc_dur)

            self.log("loss_disc_all", loss_disc_all)
            return loss_disc_all

    def validation_step(self, batch: Batch, batch_idx: int):
        val_loss = self.training_step_g(batch) + self.training_step_d(batch)
        self.log("val_loss", val_loss)
        self._log_audio_samples(self._audio_sample_dataset)
        return val_loss

    def test_step(self, batch: Batch, batch_idx: int):
        """Runs once, after training finishes (see train.py's trainer.test()
        call) -- on data the training process never touched for monitoring
        or tuning, unlike val_loss above which is checked throughout
        training and can indirectly influence decisions (e.g. when to stop).
        """
        test_loss = self.training_step_g(batch) + self.training_step_d(batch)
        self.log("test_loss", test_loss)
        return test_loss

    def _log_audio_samples(self, dataset) -> None:
        for utt_idx, utt in enumerate(dataset):
            text = utt.phoneme_ids.unsqueeze(0).to(self.device)
            text_lengths = torch.LongTensor([len(utt.phoneme_ids)]).to(self.device)
            scales = [0.667, 1.0, 0.8]
            audio = self(text, text_lengths, scales).detach()
            audio = audio * (1.0 / max(0.01, abs(audio.max())))

            tag = utt.text or str(utt_idx)
            self.logger.experiment.add_audio(tag, audio, sample_rate=self.hparams.sample_rate)

    def configure_optimizers(self):
        discriminators = [self.model_d, self.model_d_mrd, self.model_d_dur]
        discriminator_params = itertools.chain(
            *(d.parameters() for d in discriminators if d is not None)
        )
        optimizers = [
            torch.optim.AdamW(
                self.model_g.parameters(), lr=self.hparams.learning_rate,
                betas=self.hparams.betas, eps=self.hparams.eps,
            ),
            torch.optim.AdamW(
                discriminator_params, lr=self.hparams.learning_rate,
                betas=self.hparams.betas, eps=self.hparams.eps,
            ),
        ]
        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(optimizers[0], gamma=self.hparams.lr_decay),
            torch.optim.lr_scheduler.ExponentialLR(optimizers[1], gamma=self.hparams.lr_decay),
        ]
        return optimizers, schedulers

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("VitsModel")
        parser.add_argument("--batch-size", type=int, required=True)
        parser.add_argument(
            "--num-workers", type=int, default=1,
            help="Dataloader worker processes -- pure data-loading parallelism, no effect on model quality",
        )
        parser.add_argument(
            "--num-val-examples", type=int, default=500,
            help="Utterances checked periodically during training (can indirectly influence when you stop)",
        )
        parser.add_argument(
            "--num-test-examples", type=int, default=500,
            help="Utterances held out entirely until trainer.test() runs after training finishes -- never touched during training",
        )
        parser.add_argument(
            "--num-audio-samples", type=int, default=5,
            help="How many *validation* utterances to synthesize and log as listenable audio during training (not a metric)",
        )
        parser.add_argument(
            "--max-phoneme-ids", type=int,
            help="Exclude utterances with phoneme id lists longer than this",
        )
        parser.add_argument("--hidden-channels", type=int, default=192)
        parser.add_argument("--inter-channels", type=int, default=192)
        parser.add_argument("--filter-channels", type=int, default=768)
        parser.add_argument("--n-layers", type=int, default=6)
        parser.add_argument("--n-heads", type=int, default=2)
        parser.add_argument("--c-mel", type=int, default=45)
        parser.add_argument("--c-kl", type=float, default=1.0)
        parser.add_argument(
            "--posterior-encoder-layers", type=int, default=12,
            help="PosteriorEncoder WaveNet depth -- training-only cost, no effect on the exported model",
        )
        parser.add_argument("--posterior-encoder-kernel-size", type=int, default=5)
        parser.add_argument("--posterior-encoder-dilation-rate", type=int, default=1)
        parser.add_argument(
            "--flow-n-flows", type=int, default=4,
            help="Depth of the normalizing-flow stack -- part of the exported model, affects real inference cost",
        )
        parser.add_argument("--flow-kernel-size", type=int, default=5)
        parser.add_argument("--flow-dilation-rate", type=int, default=1)
        parser.add_argument(
            "--use-transformer-flows", action=argparse.BooleanOptionalAction, default=True,
            help="VITS2-style self-attention in each flow step (--no-use-transformer-flows for the lighter vanilla-VITS WN-only flow)",
        )
        parser.add_argument(
            "--use-noise-scaled-mas", action=argparse.BooleanOptionalAction, default=False,
            help="VITS2-style noise injected into the alignment search, annealed to 0 over training -- training-only, no effect on the exported model",
        )
        parser.add_argument("--mas-noise-scale-initial", type=float, default=0.01)
        parser.add_argument("--mas-noise-scale-decay", type=float, default=2e-6)
        parser.add_argument(
            "--use-snake-beta", action=argparse.BooleanOptionalAction, default=True,
            help="SnakeBeta (separate learnable alpha/beta) instead of EdgeTTS's plain Snake1d in the Generator -- BanhmiTTS-only addition, part of the exported model",
        )
        parser.add_argument(
            "--use-mrd", action=argparse.BooleanOptionalAction, default=True,
            help="UnivNet-style multi-resolution STFT discriminator -- training-only, no effect on the exported model",
        )
        parser.add_argument(
            "--use-duration-discriminator", action=argparse.BooleanOptionalAction, default=True,
            help="VITS2-style duration discriminator -- training-only, no effect on the exported model",
        )
        return parent_parser
