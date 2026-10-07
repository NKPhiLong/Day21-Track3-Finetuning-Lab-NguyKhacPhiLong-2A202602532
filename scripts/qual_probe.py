"""Per-item side-by-side of baseline (b) vs the `correct` fine-tune, for REPORT.md §6.

NB5 saves only the fine-tune's (truncated) target predictions and no regression
outputs, so the report cannot show *where* the fine-tune loses to (b). This loads the
base once, wraps it with adapters/correct, and decodes the same prompts twice: adapter
disabled = (b) with OPTIMIZED_PROMPT, adapter enabled = (c) with NAIVE_PROMPT — the
exact prompts NB2/NB5 used. Writes results/qualitative_probe.json. Reads nothing it
could change; the frozen baselines and verdict are untouched.
"""
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from peft import PeftModel

from labkit import evaluate as ev, generate, report
from labkit.config import get_tier

TIER = get_tier(os.environ.get("COMPUTE_TIER", "T4"))
TARGET_IDX = [3, 5, 12, 39, 41, 46, 0, 2, 47, 48]


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


target = load_jsonl(ROOT / "data" / "eval_target.jsonl")
regression = load_jsonl(ROOT / "data" / "eval_regression.jsonl")
tgt_rows = [target[i] for i in TARGET_IDX]

model, tok = generate.load_base(TIER)
model = PeftModel.from_pretrained(model, str(ROOT / "adapters" / "correct"))
model.eval()


def run(system_target):
    tp, _ = generate.generate_batch(model, tok, [r["input"] for r in tgt_rows],
                                    system=system_target, label="target")
    rp, _ = generate.generate_batch(model, tok, [r["instruction"] for r in regression],
                                    system=None, max_new_tokens=96, label="regression")
    return tp, rp


with model.disable_adapter():
    b_t, b_r = run(generate.OPTIMIZED_PROMPT)
c_t, c_r = run(generate.NAIVE_PROMPT)

out = {"target": [], "regression": []}
for i, r, pb, pc in zip(TARGET_IDX, tgt_rows, b_t, c_t):
    out["target"].append({"i": i, "ticket": r["input"], "label": r["label"],
                          "b_pred": pb.strip(), "b_score": ev.triage_field_accuracy(pb, r["label"]),
                          "c_pred": pc.strip(), "c_score": ev.triage_field_accuracy(pc, r["label"])})
for j, (r, pb, pc) in enumerate(zip(regression, b_r, c_r)):
    out["regression"].append({"j": j, "question": r["instruction"], "keywords": r["keywords"],
                              "b_pred": pb.strip(), "b_recall": ev.keyword_recall(pb, r["keywords"]),
                              "c_pred": pc.strip(), "c_recall": ev.keyword_recall(pc, r["keywords"])})

n = len(regression)
out["regression_mean"] = {"b": sum(x["b_recall"] for x in out["regression"]) / n,
                          "c": sum(x["c_recall"] for x in out["regression"]) / n}
report.write_json(out, "qualitative_probe.json", results_dir=ROOT / "results")
print(json.dumps(out["regression_mean"]))
for x in out["target"]:
    print("T", x["i"], x["b_score"], x["c_score"])
for x in out["regression"]:
    print("R", x["j"], round(x["b_recall"], 3), round(x["c_recall"], 3))
