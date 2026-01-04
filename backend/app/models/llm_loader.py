# app/models/llm_loader.py

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from app.utils.device import detect_device
from app.config import HF_TOKEN

# Using Microsoft Phi-2 as it is freely accessible without gated access
TCS_MODEL_NAME = "microsoft/phi-2"

_tokenizer = None
_model = None


def load_tcs_model():
    global _tokenizer, _model

    if _model is not None:
        return _tokenizer, _model

    # Try to get HF_TOKEN from environment, then from config
    hf_token = os.getenv("HF_TOKEN", HF_TOKEN)
    if not hf_token:
        raise RuntimeError("HF_TOKEN not configured in config.py or environment")

    device = detect_device()

    _tokenizer = AutoTokenizer.from_pretrained(
        TCS_MODEL_NAME,
        use_fast=True,
        token=hf_token
    )

    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    if device == "cuda":
        dtype = torch.float16
        device_map = "auto"
    else:
        dtype = torch.float32
        device_map = None

    _model = AutoModelForCausalLM.from_pretrained(
        TCS_MODEL_NAME,
        torch_dtype=dtype,
        device_map=device_map,
        token=hf_token
    )

    _model.eval()
    torch.set_grad_enabled(False)

    return _tokenizer, _model
