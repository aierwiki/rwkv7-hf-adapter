#!/usr/bin/env python3
# coding=utf-8
"""Correctness smoke for the native/no-FLA RWKV-7 code path (``RWKV7_NATIVE_MODEL=1``).

Verifies that the native path produces a deterministic greedy continuation
(``[0..15] → 16``) — not just runs without crashing.  This guards the pure-PyTorch
path used on ROCm / no-GPU / CPU-only setups.

Run as::

    RWKV7_TEST_MODEL=rwkv7-g1d-0.1b-hf pytest tests/test_native_path.py -v
    RWKV7_NATIVE_MODEL=1 python tests/test_native_path.py --model rwkv7-g1d-0.1b-hf
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    # ROCm exposes itself as CUDA via PyTorch; explicit HIP check as fallback.
    try:
        if torch.version.hip is not None:
            return "cuda"
    except Exception:
        pass
    return "cpu"


_MODEL: str = os.environ.get("RWKV7_TEST_MODEL", "rwkv7-g1d-0.1b-hf")
_DEV: str = _resolve_device()


def test_native_path_greedy_continuation() -> None:
    """Greedy forward: input [0..15] must output token 16 under native path."""
    from transformers import AutoModelForCausalLM

    m = (
        AutoModelForCausalLM.from_pretrained(
            _MODEL, torch_dtype=torch.float16, trust_remote_code=True
        )
        .to(_DEV)
        .eval()
    )
    ids = torch.tensor([list(range(16))], device=_DEV)
    with torch.no_grad():
        nxt = m(ids).logits[0, -1].argmax().item()
    assert nxt == 16, f"native path greedy [0..15] -> {nxt}, expected 16"


def main() -> int:
    global _MODEL, _DEV

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=_MODEL)
    ap.add_argument("--device", default=_DEV)
    args = ap.parse_args()

    # Hook globals so the pytest test picks them up from CLI args.
    _MODEL = args.model
    _DEV = args.device

    # Ensure native path is active when run standalone.
    os.environ.setdefault("RWKV7_NATIVE_MODEL", "1")

    print(f"model: {_MODEL}  device: {_DEV}")
    test_native_path_greedy_continuation()
    print("PASS: native path greedy [0..15] -> 16")
    return 0


if __name__ == "__main__":
    sys.exit(main())
