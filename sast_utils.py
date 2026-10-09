"""
SAST Utilities - Backward-compatibility shim
=============================================
Code has been split into focused modules:
  sast_config.py     - SastConfig + set_seed
  sast_data.py       - load_dataset
  sast_model.py      - create_model, create_freq_encoder, get_freq_bins, load_checkpoint
  sast_inference.py  - infer_sast, predict_class, compute_class_centroids, evaluate_accuracy
  sast_viz.py        - plot_tfr, plot_full_diagnostics, print_inference_summary

This shim re-exports everything so existing `from sast_utils import X` still works.
New code should import from the specific submodule instead.
"""
from sast_config import SastConfig, set_seed
from sast_data import load_dataset
from sast_model import create_model, create_freq_encoder, get_freq_bins, load_checkpoint
from sast_inference import (
    infer_sast, predict_class, compute_class_centroids, evaluate_accuracy,
)
from sast_viz import (
    plot_tfr, plot_full_diagnostics, print_inference_summary,
)

__all__ = [
    # config
    'SastConfig', 'set_seed',
    # data
    'load_dataset',
    # model
    'create_model', 'create_freq_encoder', 'get_freq_bins', 'load_checkpoint',
    # inference
    'infer_sast', 'predict_class', 'compute_class_centroids', 'evaluate_accuracy',
    # viz
    'plot_tfr', 'plot_full_diagnostics', 'print_inference_summary',
]
