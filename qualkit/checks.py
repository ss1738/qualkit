"""Core pre-flight assurance checks for compressed classification models.

Every check implements a measured finding from the assured-EO research
programme (github.com/ss1738/euroSAT-CNN-landcover). All checks are
model-agnostic: they need only a torch.nn.Module and a calibration
DataLoader yielding (input, label) batches.
"""
import copy
import hashlib
import random
import time

import torch
import torch.nn as nn


@torch.no_grad()
def _softmaxes(model, loader, device):
    model = model.to(device).eval()
    ps, ys = [], []
    for x, y in loader:
        ps.append(torch.softmax(model(x.to(device)), dim=1).cpu())
        ys.append(y)
    return torch.cat(ps), torch.cat(ys)


def _scores(p, y):
    return 1.0 - p[torch.arange(len(y)), y]


def _qhat(scores, alpha):
    k = min(int((1 - alpha) * (len(scores) + 1)) / len(scores), 1.0)
    return torch.quantile(scores, k).item()


def _tv(a, b, bins=1000):
    ha = torch.histc(a, bins, 0, 1) / len(a)
    hb = torch.histc(b, bins, 0, 1) / len(b)
    return 0.5 * (ha - hb).abs().sum().item()


def coverage_forecast(model_ref, model_deploy, cal_loader, alpha=0.10, device="cpu"):
    """Finding 9/10: coverage loss under stale calibration is upper-bounded by
    (and on peaked tasks approximately equals) the TV distance between the
    reference and deployment score distributions on the calibration set.
    Both distributions are computable on the ground before flight."""
    p_ref, y = _softmaxes(model_ref, cal_loader, device)
    p_dep, _ = _softmaxes(model_deploy, cal_loader, device)
    s_ref, s_dep = _scores(p_ref, y), _scores(p_dep, y)
    tv = _tv(s_ref, s_dep)
    qh_ref = _qhat(s_ref, alpha)
    cov_stale = (s_dep <= qh_ref).float().mean().item()
    return {
        "alpha": alpha,
        "target_coverage": 1 - alpha,
        "tv_distance": tv,
        "forecast_coverage_floor": 1 - alpha - tv,
        "measured_stale_coverage_on_cal": cov_stale,
        "verdict": "RECALIBRATE" if tv > 0.01 else "STALE_CAL_OK",
        "note": "Coverage after deployment with stale calibration will not fall "
                "below the forecast floor; on peaked score distributions the "
                "deficit approximately equals the TV distance.",
    }


def setsize_acceptance(model_deploy, cal_loader, alpha=0.10, max_mean_setsize=1.5,
                       device="cpu"):
    """Finding 11: after recalibration, validity always returns but prediction
    sets can balloon (the efficiency knee). Mean set size on calibration data
    is the acceptance test for whether the compressed model is worth flying."""
    p, y = _softmaxes(model_deploy, cal_loader, device)
    s = _scores(p, y)
    qh = _qhat(s, alpha)
    sets = p >= (1 - qh)
    mean_size = sets.sum(1).float().mean().item()
    empty = (sets.sum(1) == 0).float().mean().item()
    return {
        "alpha": alpha,
        "recalibrated_qhat": qh,
        "mean_set_size": mean_size,
        "empty_set_rate": empty,
        "threshold": max_mean_setsize,
        "verdict": "PASS" if mean_size <= max_mean_setsize else "FAIL",
        "note": "FAIL means the model is valid but uninformative at this "
                "precision; use a higher bit-width or a stronger model.",
    }


def seu_probe(model, probe_loader, trials=20, bits=None, device="cpu", seed=0):
    """Finding 12: single stored-weight bit flips are 83-100% output-silent, so
    no output-side monitor can count faults. Integer-quantized storage bounds
    per-flip damage; FP32 does not. This probe measures the silent fraction
    and worst-case accuracy drop for single flips on THIS model."""
    base = copy.deepcopy(model).to(device).eval()
    with torch.no_grad():
        xs, ys = next(iter(probe_loader))
        xs = xs.to(device)
        pred0 = base(xs).argmax(1).cpu()
    params = [p for p in base.parameters() if p.numel() > 0]
    sizes = [p.numel() for p in params]
    total = sum(sizes)
    silent = 0
    worst_change = 0.0
    rng = random.Random(seed)
    for t in range(trials):
        m = copy.deepcopy(base)
        ps = [p for p in m.parameters() if p.numel() > 0]
        gi = rng.randrange(total)
        with torch.no_grad():
            for p, sz in zip(ps, sizes):
                if gi < sz:
                    if bits is None:  # FP32 flip
                        v = p.data.view(-1)[gi:gi + 1].view(torch.int32)
                        v ^= (1 << rng.randrange(32))
                    else:  # integer-code flip, magnitude-bounded
                        w = p.data.view(-1)
                        qm = 2 ** (bits - 1) - 1
                        sc = w.abs().max().clamp(min=1e-8) / qm
                        code = int(round((w[gi] / sc).item()))
                        code = (code & (2 ** bits - 1)) ^ (1 << rng.randrange(bits))
                        if code >= 2 ** (bits - 1):
                            code -= 2 ** bits
                        w[gi] = max(-qm - 1, min(qm, code)) * sc
                    break
                gi -= sz
            pred = m(xs).argmax(1).cpu()
        changed = (pred != pred0).float().mean().item()
        if changed == 0:
            silent += 1
        worst_change = max(worst_change, changed)
    return {
        "trials": trials,
        "storage_format": "fp32" if bits is None else f"int{bits}",
        "silent_fraction": silent / trials,
        "worst_single_flip_prediction_change": worst_change,
        "note": "High silent fraction means output monitoring cannot detect "
                "faults; pair with the integrity check below.",
    }


def integrity_baseline(model, probe_loader=None, device="cpu"):
    """Finding 12 fix: weight-side integrity via hashing. Reports the reference
    digest and the measured cost of one hash relative to one inference, so a
    mission can pick its hash-every-N-frames overhead."""
    m = copy.deepcopy(model).cpu().eval()
    wbytes = b"".join(p.data.numpy().tobytes() for p in m.parameters())
    t0 = time.time()
    digest = hashlib.sha256(wbytes).hexdigest()
    hash_s = time.time() - t0
    infer_s = None
    if probe_loader is not None:
        xs, _ = next(iter(probe_loader))
        with torch.no_grad():
            m(xs[:1])
            t0 = time.time()
            for _ in range(10):
                m(xs[:1])
            infer_s = (time.time() - t0) / 10
    return {
        "sha256": digest,
        "weight_bytes": len(wbytes),
        "hash_seconds": hash_s,
        "inference_seconds": infer_s,
        "hash_to_inference_ratio": (hash_s / infer_s) if infer_s else None,
        "note": "Verify the digest periodically; hashing every N frames costs "
                "ratio/N of inference time (e.g. ratio 4.3, N=100 -> ~4%).",
    }
