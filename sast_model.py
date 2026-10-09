"""
SAST Model Factory
==================
create_model, create_freq_encoder, get_freq_bins, load_checkpoint.
"""
from typing import Optional, Tuple

import numpy as np
import torch

from models.sast import SAST
from models.sast_losses import FreqEncoder
from models.sast_graph import get_graph_summary
from sast_config import SastConfig


def create_model(config: SastConfig, device: torch.device) -> SAST:
    """Create a fresh SAST model from config."""
    model = SAST(
        fs=config.fs,
        d_h=config.d_h,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        sigma_min=config.sigma_min,
        sigma_max=config.sigma_max,
        msst_num=config.msst_num,
        d_cond=config.d_cond,
        f_type_embed_dim=config.f_type_embed_dim,
        ppn_temperature=config.ppn_temperature,
        prototype_temperature=config.prototype_temperature,
        dropout=config.dropout,
        n_sqz_max=config.n_sqz_max,
    ).to(device)
    return model


def create_freq_encoder(n_freq_bins: int,
                        embed_dim: int = 128,
                        device: Optional[torch.device] = None) -> FreqEncoder:
    """Create a FreqEncoder (TFR -> z_freq) for SupCon."""
    enc = FreqEncoder(n_freq_bins=n_freq_bins, embed_dim=embed_dim)
    if device is not None:
        enc = enc.to(device)
    return enc


def get_freq_bins(model: SAST, sample_signal: np.ndarray,
                  device: torch.device) -> int:
    """Run a dummy forward pass to determine F_bins from model output."""
    dummy = torch.from_numpy(sample_signal).float().unsqueeze(0).to(device)
    with torch.no_grad():
        result = model(dummy, return_all=True)
    return result['freqs'].shape[0]


def load_checkpoint(checkpoint_path: str,
                    device: torch.device
                    ) -> Tuple[SAST, FreqEncoder, dict]:
    """
    Load model + freq_encoder from a training checkpoint.

    Returns:
        model, freq_encoder, metadata (dict with config + C_prior)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get('args', {})

    config = SastConfig.from_dict(ckpt_args)

    model = create_model(config, device)
    model_missing, model_unexpected = model.load_state_dict(
        ckpt['model_state_dict'], strict=False)
    if model_missing:
        print(f"  [load_checkpoint] model missing keys (用随机初值): {model_missing}")
    if model_unexpected:
        print(f"  [load_checkpoint] model unexpected keys: {model_unexpected}")

    # Infer F_bins from encoder state dict
    enc_weight = ckpt['encoder_state_dict']['encoder.0.weight']
    F_bins = enc_weight.shape[1]

    freq_encoder = create_freq_encoder(F_bins, device=device)
    enc_missing, enc_unexpected = freq_encoder.load_state_dict(
        ckpt['encoder_state_dict'], strict=False)
    if enc_missing:
        print(f"  [load_checkpoint] encoder missing keys: {enc_missing}")
    if enc_unexpected:
        print(f"  [load_checkpoint] encoder unexpected keys: {enc_unexpected}")

    model.eval()
    freq_encoder.eval()

    C_prior = model.get_C_prior().cpu().numpy()

    print(f"Loaded SAST checkpoint: {checkpoint_path}")
    print(f"  d_h={config.d_h}, n_heads={config.n_heads}, n_layers={config.n_layers}")
    print(f"  fs={config.fs} Hz, F_bins={F_bins}")
    print(f"  C_prior: {C_prior}")
    print(get_graph_summary())

    metadata = {
        'config': config,
        'C_prior': C_prior,
        'F_bins': F_bins,
        'checkpoint_args': ckpt_args,
    }
    return model, freq_encoder, metadata
