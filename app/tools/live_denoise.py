"""Conservative block denoising for the realtime path.

The caller owns one instance per stream (input and output need separate
instances).  A voice block is never used as the noise reference.  Until a
quiet reference has been collected, processing is a lossless bypass.
"""

import torch

from tools.torchgate import TorchGate


class LiveDenoiser:
    """Learn noise only from quiet blocks and apply a bounded spectral gate."""

    def __init__(self, sr, n_fft, *, quiet_threshold_db=-45.0,
                 quiet_seconds=0.08, reference_seconds=2.0,
                 max_attenuation_db=8.0, device=None):
        if sr <= 0 or n_fft <= 0:
            raise ValueError("sr and n_fft must be positive")
        if max_attenuation_db < 0 or max_attenuation_db > 10:
            raise ValueError("max_attenuation_db must be between 0 and 10")
        self.sr = int(sr)
        self.n_fft = int(n_fft)
        self.quiet_threshold_db = float(quiet_threshold_db)
        self.min_quiet_samples = max(self.n_fft * 2, int(round(sr * quiet_seconds)))
        self.max_reference_samples = max(self.min_quiet_samples,
                                         int(round(sr * reference_seconds)))
        self._quiet = []
        self._quiet_samples = 0
        self._reference_cache = None
        self.frame_samples = max(1, int(round(sr * 0.04)))
        self.gate = TorchGate(
            sr=self.sr,
            n_fft=self.n_fft,
            prop_decrease=1.0 - 10.0 ** (-float(max_attenuation_db) / 20.0),
        ).to(device=device) if device is not None else TorchGate(
            sr=self.sr,
            n_fft=self.n_fft,
            prop_decrease=1.0 - 10.0 ** (-float(max_attenuation_db) / 20.0),
        )

    @property
    def ready(self):
        return self._quiet_samples >= self.min_quiet_samples

    def reset(self):
        self._quiet.clear()
        self._quiet_samples = 0
        self._reference_cache = None

    def _rms_db(self, audio):
        rms = torch.sqrt(torch.mean(audio.float() ** 2)).item()
        return 20.0 * torch.log10(torch.tensor(max(rms, 1e-7))).item()

    def _remember_quiet(self, audio):
        # The caller's realtime ring buffer is mutated on the next callback.
        chunk = audio.detach().float().reshape(-1).clone()
        self._quiet.append(chunk)
        self._quiet_samples += chunk.numel()
        while self._quiet_samples > self.max_reference_samples:
            removed = self._quiet.pop(0)
            self._quiet_samples -= removed.numel()

    def _quiet_candidates(self, audio):
        count = audio.numel() // self.frame_samples
        if count == 0:
            return torch.empty(0, dtype=torch.long)
        frames = audio[:count * self.frame_samples].reshape(count, self.frame_samples)
        power = frames.float().square()
        rms = torch.sqrt(power.mean(dim=1))
        level_db = 20.0 * torch.log10(rms.clamp_min(1e-7))
        spectrum_power = torch.fft.rfft(frames.float(), dim=1).abs().square()
        spectrum_power = spectrum_power[:, 1:].clamp_min(1e-12)
        flatness = spectrum_power.log().mean(dim=1).exp() / spectrum_power.mean(dim=1)
        candidates = (rms > 1e-7) & (level_db <= self.quiet_threshold_db) & (flatness > 0.2)
        # One device-to-host transfer for all candidate indices, rather than
        # synchronizing once per frame on .item().
        return torch.nonzero(candidates, as_tuple=False).flatten().cpu()

    def _reference(self, device, dtype):
        if self._reference_cache is None:
            return None
        if self._reference_cache.device != device:
            self._reference_cache = self._reference_cache.to(device=device)
        return self._reference_cache.to(dtype=dtype)

    @torch.no_grad()
    def process(self, audio):
        """Process one mono block and preserve its shape/device/dtype."""
        if not torch.is_tensor(audio):
            raise TypeError("audio must be a torch.Tensor")
        original_shape = audio.shape
        flat = audio.reshape(-1)
        if flat.numel() < self.n_fft:
            return audio
        candidates = self._quiet_candidates(flat)
        learned_this_block = bool(candidates.numel())
        for index in candidates.tolist():
            start = index * self.frame_samples
            self._remember_quiet(flat[start:start + self.frame_samples])
        if learned_this_block:
            self._reference_cache = torch.cat(tuple(self._quiet), dim=0)
        if (not learned_this_block and
                self._rms_db(flat) <= self.quiet_threshold_db):
            return audio
        if not self.ready:
            return audio
        if self.gate.stft_window.device != flat.device:
            self.gate.to(device=flat.device)
        reference = self._reference(flat.device, flat.dtype)
        result = self.gate(flat.unsqueeze(0), reference.unsqueeze(0)).squeeze(0)
        # TorchGate's bounded prop_decrease limits spectral attenuation.  Keep
        # shape and caller dtype exactly as supplied.
        return result.reshape(original_shape).to(dtype=audio.dtype)
