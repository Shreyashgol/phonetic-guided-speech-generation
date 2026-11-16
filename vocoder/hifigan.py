"""Pretrained HiFi-GAN (SpeechBrain, trained on LJSpeech at 22.05 kHz, 80 mels, hop 256)."""
import torch


class Vocoder:
    def __init__(self, device='cpu', savedir='pretrained_hifigan'):
        try:
            from speechbrain.inference.vocoders import HIFIGAN
        except ImportError:                                   # older speechbrain
            from speechbrain.pretrained import HIFIGAN
        self.device = device
        self.m = HIFIGAN.from_hparams(source='speechbrain/tts-hifigan-ljspeech', savedir=savedir, run_opts={'device': device})

    @torch.no_grad()
    def __call__(self, mel):
        """mel: (B, T, 80) or (T, 80) log-mel in natural units (NOT normalised) -> waveform (B, T*256) float tensor on CPU."""
        if mel.dim() == 2: mel = mel.unsqueeze(0)
        w = self.m.decode_batch(mel.transpose(1, 2).float().to(self.device))
        return w.reshape(w.shape[0], -1).cpu()
