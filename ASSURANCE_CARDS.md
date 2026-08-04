# Assurance Cards (spec AC-0.1)

Model cards describe a model. An Assurance Card states what has been
**measured** about the model's reliability, under which assumptions those
measurements are valid, and how to detect that the artifact you are running is
the one the card was issued for. It is generated from test results (a qualkit
report), never hand-written, so every number on it is reproducible.

## Why a new artifact

Model cards and AI FactSheets are descriptive documentation. Certification
regimes (space: ECSS-E-HB-40-02A; aviation: EASA learning assurance) and the
EU AI Act ask for evidence. Nothing in between exists: a compact, versioned,
machine-readable artifact carrying finite-sample guarantees. That is this.

## Required fields (AC-0.1)

| Block | Field | Meaning |
|---|---|---|
| model | name, version, dataset | identity |
| model | weights_sha256, weight_bytes | binds the card to ONE artifact |
| guarantee | type, alpha, target_coverage | the statistical guarantee claimed |
| guarantee | recalibrated_qhat | the deployed decision threshold |
| guarantee | valid_if | the assumptions; the card is void outside them |
| compression | tv_distance_vs_reference | measured score-distribution shift |
| compression | stale_calibration_coverage_floor | forecast coverage if calibration is not refreshed (validated bound: held 60/60 runs across 2 datasets, 2 architectures, 5 bit-widths) |
| compression | verdict | RECALIBRATE or STALE_CAL_OK |
| efficiency | mean_prediction_set_size, empty_set_rate, verdict | informativeness after recalibration |
| fault_exposure | storage_format, single_flip_output_silent_fraction | measured SEU susceptibility; silent faults require weight-side integrity checks |
| issued, notes, generated_by | provenance |

## Rules

1. Cards are generated, not written. Every number must come from an executed
   check.
2. A card binds to exactly one weights digest. New weights, new card.
3. The `valid_if` assumptions are part of the guarantee. Quoting the coverage
   number without them is misuse of the card.
4. Cards state failure modes (fault exposure, stale-calibration floor), not
   only strengths. A card with no stated weakness is not an assurance card.

## Generate one

```python
from qualkit import report, card
r = report.run_all(model_ref, model_deploy, cal_loader, probe_loader)
c = card.make_card(r, model_name="landcover-int8", model_version="1.2.0",
                   dataset="EuroSAT RGB", issued="2026-08-04")
open("ASSURANCE_CARD.md", "w").write(card.card_markdown(c))
```

Research behind every field: github.com/ss1738/euroSAT-CNN-landcover
