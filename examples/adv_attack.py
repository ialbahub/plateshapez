#!/usr/bin/env python3
"""Adversarial (gradient) attack on EasyOCR's recognizer — subtle by design.

Hand-tuned marks can't beat OCR without becoming visible. A gradient-based
adversarial perturbation can: it nudges pixels in the direction that maximises
the recognizer's loss for the true label, bounded by a small L-infinity epsilon
so the change stays faint. This script proves the principle on EasyOCR's CRNN:
clean read -> PGD attack at several epsilons -> adversarial read, reporting the
per-pixel budget so you can see how faint it is.

Run with: uv run python examples/adv_attack.py
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

from plateshapez.synthetic import create_plate_overlay

PLATE = "Q0D1I5"
IMG_H = 64  # EasyOCR recognition height


def to_input(gray: Image.Image) -> torch.Tensor:
    """Grayscale PIL -> normalized recognizer tensor [1,1,H,W] in [-1, 1]."""
    w, h = gray.size
    new_w = max(IMG_H, int(w * IMG_H / h))
    arr = np.asarray(gray.resize((new_w, IMG_H)), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return (t - 0.5) / 0.5


def decode(reader: Any, logits: torch.Tensor) -> str:
    _, idx = logits.max(2)
    flat = idx.view(-1).detach().cpu().numpy()
    return reader.converter.decode_greedy(flat, [logits.size(1)])[0]


def main() -> None:
    import easyocr

    # quantize=False: the default CPU model is dynamically quantized (non-
    # differentiable LSTM/Linear), which severs gradients to the input.
    reader = easyocr.Reader(["en"], gpu=False, verbose=False, quantize=False)
    model = reader.recognizer.eval()  # params stay trainable so grad flows to input

    # Clean character-band crop of the plate on a dark background.
    plate = create_plate_overlay(PLATE)
    flat = Image.new("RGB", plate.size, (18, 18, 20))
    flat.paste(plate, (0, 0), plate)
    w, h = plate.size
    band = flat.crop((int(w * 0.05), int(h * 0.30), int(w * 0.95), int(h * 0.72)))
    gray = band.convert("L")

    x0 = to_input(gray)
    dummy = torch.zeros(1, 40, dtype=torch.long)
    target, lengths = reader.converter.encode([PLATE], batch_max_length=25)
    ctc = torch.nn.CTCLoss(zero_infinity=True)

    clean_read = decode(reader, model(x0, dummy).softmax(2))
    print(f"clean recognizer read: {clean_read!r}")

    # sanity: gradient must flow from the CTC loss to the input
    xc = x0.clone().requires_grad_(True)
    preds = model(xc, dummy)
    logp = preds.log_softmax(2).permute(1, 0, 2)
    loss0 = ctc(logp, target, torch.IntTensor([preds.size(1)]), lengths)
    g0 = torch.autograd.grad(loss0, xc)[0]
    print(f"clean CTC loss {float(loss0):.3f}; grad max {float(g0.abs().max()):.4f}")

    # Targeted PGD: drive the read toward a string far from the truth.
    tgt_text = "8B8B8B"
    t_tgt, t_len = reader.converter.encode([tgt_text], batch_max_length=25)

    print(f"\ntargeted toward {tgt_text!r}; untargeted maximises true-label loss")
    print(f"{'eps(px)':>8} {'targeted read':>16} {'untargeted read':>16} {'mean|d|px':>11}")
    for eps_px in (2, 4, 8, 16, 32):
        eps = eps_px * 2 / 255
        rows = []
        for mode in ("targeted", "untargeted"):
            x_adv = (x0 + torch.empty_like(x0).uniform_(-eps, eps)).clamp(-1, 1)
            for _ in range(150):
                x_adv = x_adv.detach().requires_grad_(True)
                preds = model(x_adv, dummy)
                logp = preds.log_softmax(2).permute(1, 0, 2)
                psize = torch.IntTensor([preds.size(1)])
                if mode == "targeted":
                    loss = ctc(logp, t_tgt, psize, t_len)  # minimise => become target
                    direction = -1.0
                else:
                    loss = ctc(logp, target, psize, lengths)  # maximise => unread truth
                    direction = 1.0
                grad = torch.autograd.grad(loss, x_adv)[0]
                with torch.no_grad():
                    x_adv = x_adv + direction * (eps / 10) * grad.sign()
                    x_adv = torch.min(torch.max(x_adv, x0 - eps), x0 + eps).clamp(-1, 1)
            rows.append((decode(reader, model(x_adv, dummy).softmax(2)),
                         float((x_adv - x0).abs().mean()) * 255 / 2))
        print(f"{eps_px:>8} {rows[0][0] or '∅':>16} {rows[1][0] or '∅':>16} {rows[0][1]:>11.2f}")


if __name__ == "__main__":
    main()
