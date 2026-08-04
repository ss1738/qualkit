"""Run all checks and emit a pre-flight assurance report (dict + markdown)."""
import json

from . import checks


def run_all(model_ref, model_deploy, cal_loader, probe_loader=None,
            alpha=0.10, seu_bits=8, device="cpu"):
    r = {}
    r["coverage_forecast"] = checks.coverage_forecast(
        model_ref, model_deploy, cal_loader, alpha=alpha, device=device)
    r["setsize_acceptance"] = checks.setsize_acceptance(
        model_deploy, cal_loader, alpha=alpha, device=device)
    if probe_loader is not None:
        r["seu_probe"] = checks.seu_probe(
            model_deploy, probe_loader, bits=seu_bits, device=device)
        r["integrity"] = checks.integrity_baseline(model_deploy, probe_loader)
    else:
        r["integrity"] = checks.integrity_baseline(model_deploy)
    return r


def to_markdown(r):
    lines = ["# qualkit pre-flight assurance report", ""]
    cf = r["coverage_forecast"]
    lines += [
        "## 1. Coverage forecast (stale-calibration risk)",
        f"- TV distance between reference and deployment scores: **{cf['tv_distance']:.4f}**",
        f"- Forecast coverage floor with stale calibration: **{cf['forecast_coverage_floor']:.3f}** "
        f"(target {cf['target_coverage']:.2f})",
        f"- Verdict: **{cf['verdict']}**",
        "",
    ]
    ss = r["setsize_acceptance"]
    lines += [
        "## 2. Set-size acceptance (informativeness after recalibration)",
        f"- Mean prediction-set size on calibration data: **{ss['mean_set_size']:.2f}** "
        f"(threshold {ss['threshold']})",
        f"- Empty-set (refusal) rate: {ss['empty_set_rate']:.3f}",
        f"- Verdict: **{ss['verdict']}**",
        "",
    ]
    if "seu_probe" in r:
        sp = r["seu_probe"]
        lines += [
            "## 3. SEU probe (single stored-weight bit flips)",
            f"- Storage format: {sp['storage_format']}",
            f"- Output-silent fraction: **{sp['silent_fraction']:.2f}** "
            "(silent faults are invisible to any output-side monitor)",
            f"- Worst single-flip prediction change: {sp['worst_single_flip_prediction_change']:.3f}",
            "",
        ]
    ig = r["integrity"]
    ratio = ig["hash_to_inference_ratio"]
    lines += [
        "## 4. Weight integrity baseline",
        f"- sha256: `{ig['sha256'][:16]}...` over {ig['weight_bytes']:,} bytes",
        f"- Hash cost vs one inference: "
        + (f"**{ratio:.1f}x** (hash every 100 frames ~= {ratio:.0f}% overhead)"
           if ratio else "n/a (no probe loader given)"),
        "",
        "---",
        "Checks implement measured findings from "
        "github.com/ss1738/euroSAT-CNN-landcover (12 findings, reproducible).",
    ]
    return "\n".join(lines)


def save(r, path_json="qualkit_report.json", path_md="qualkit_report.md"):
    with open(path_json, "w") as f:
        json.dump(r, f, indent=2)
    with open(path_md, "w") as f:
        f.write(to_markdown(r))
    return path_json, path_md
