"""Manifest v2: versioned pre-outcome corrections from the Codex Gate C review
(2026-08-30). Reads parent_annotations.json (v1, left untouched) and writes
parent_annotations_v2.json. No forced outcome existed when this ran.

Corrections:
  C1. below_good_03, below_good_04, above_good_02: v1's c1 was the numbered-list
      artifact "1." (not substantive). v2 c1 = end of sentence sid=2, the first
      actual task-analysis bullet ("Estimate the total number of black spots...").
  C2. above_good_13 c3: char_end 762 -> 763 so the boundary includes the closing
      quote of the quoted trivia sentence; the cut becomes clean (160/160).
  C3. Rename cut field incentive_visible -> incentive_rementioned_in_reasoning:
      the conditioned user prompt is present in EVERY prefix, so no cut is
      incentive-free; the flag only records re-mention inside the reasoning.
  C4. Per-parent flags: c2_hedged (adoption clause carries alternative values;
      explicit 11-parent list, see C2_HEDGED) and
      c3_contains_target_calc (the c3 sentence contains a computed target
      total; exactly six cases). Both feed preregistered sensitivity analyses:
      report the c2 contrast excluding hedged parents and the c3 contrast
      excluding target-calc cells.

  uv run python -m value_leakage.fw_manifest_v2 --run_dir=runs/fw_parents_20260830_134529
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from value_leakage.fw_annotate_prep import char_to_token_boundary, load_parent, sentences

C1_FIX = {"below_good_03": 2, "below_good_04": 2, "above_good_02": 2}
# Explicit list: parents whose C2 adoption clause carries alternative values.
# (A notes-substring rule would wrongly include above_good_11, whose "hedged"
# note refers to its c3/e3 sentence; its c2 "Let's use 117,000." is unhedged.)
C2_HEDGED = {"below_good_01", "below_good_06", "below_good_07", "below_good_11",
             "below_good_13", "below_good_14", "below_good_17",
             "above_good_03", "above_good_06", "above_good_16", "above_good_17"}
C3_TARGET_CALC = {"below_good_06", "below_good_13", "below_good_16", "below_good_19",
                  "above_good_02", "above_good_12"}


def main(run_dir: str = "runs/fw_parents_20260830_134529") -> None:
    run = Path(run_dir)
    v1 = json.loads((run / "parent_annotations.json").read_text())
    assert v1.get("version", 1) == 1 and v1["annotation_status"] == "frozen_before_any_forced_outcome"
    out_path = run / "parent_annotations_v2.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; refusing to overwrite")

    v2 = json.loads(json.dumps(v1))  # deep copy
    changelog = []
    for p in v2["parents"]:
        pid = p["parent_id"]
        par = load_parent(run, pid)
        # C1
        if pid in C1_FIX:
            sents = sentences(par["reasoning"])
            sid = C1_FIX[pid]
            char_end = sents[sid]["end"]
            b = char_to_token_boundary(par, char_end)
            assert b["clean"], f"{pid} corrected c1 not on clean boundary"
            old = p["cuts"]["c1"]
            p["cuts"]["c1"] = {"missing": False, "char_end": char_end,
                               "sentence_text": sents[sid]["text"],
                               "prefix_tail_decoded": par["completion_text"][:b["boundary_char"]][-80:],
                               "incentive_visible": (p.get("incentive_first_mention_char") is not None
                                                     and p["incentive_first_mention_char"] < char_end),
                               **b}
            changelog.append(f"{pid}: c1 moved from {old['token_index']} ({old['sentence_text']!r}) "
                             f"to token {b['token_index']} (first task-analysis bullet, sid={sid})")
        # C2
        if pid == "above_good_13":
            char_end = p["cuts"]["c3"]["char_end"] + 1
            b = char_to_token_boundary(par, char_end)
            assert b["clean"] and b["token_index"] == p["cuts"]["c3"]["token_index"]
            p["cuts"]["c3"].update({"char_end": char_end,
                                    "sentence_text": p["cuts"]["c3"]["sentence_text"] + '"',
                                    **b})
            changelog.append(f"{pid}: c3 char_end +1 to include the closing quote already inside "
                             f"token boundary {b['token_index']}; cut now clean")
        # C3 rename
        for name, cut in p["cuts"].items():
            if "incentive_visible" in cut:
                cut["incentive_rementioned_in_reasoning"] = cut.pop("incentive_visible")
        # C4 flags
        p["c2_hedged"] = pid in C2_HEDGED
        p["c3_contains_target_calc"] = pid in C3_TARGET_CALC
        # ordering re-check
        order = [p["cuts"][n].get("token_index") for n in ("c1", "c2", "c3", "c4")
                 if not p["cuts"][n].get("missing")]
        assert order == sorted(order) and len(set(order)) == len(order), f"{pid} order violation"
        # target-calc flag consistency: sentence contains a >=1e6 '=' result
        import re
        has_calc = bool(re.search(r"[=s]\s*~?[\d,]{8,}", p["cuts"]["c3"]["sentence_text"] or ""))
        assert has_calc == p["c3_contains_target_calc"], \
            f"{pid}: c3_contains_target_calc={p['c3_contains_target_calc']} but sentence says otherwise: {p['cuts']['c3']['sentence_text']!r}"

    hedged = sorted(p["parent_id"] for p in v2["parents"] if p["c2_hedged"])
    clean = sum(not c.get("missing") and c.get("clean") for p in v2["parents"] for c in p["cuts"].values())
    total = sum(not c.get("missing") for p in v2["parents"] for c in p["cuts"].values())
    v2.update({
        "version": 2,
        "supersedes": "parent_annotations.json (v1, preserved unchanged)",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "v2_reason": "Codex Gate C review corrections, applied before any forced outcome",
        "v2_changelog": changelog,
        "c2_hedged_parents": hedged,
        "c3_target_calc_parents": sorted(C3_TARGET_CALC),
        "preregistered_sensitivities": [
            "E2 c2 contrast excluding c2_hedged parents",
            "E2 c3 contrast excluding c3_contains_target_calc parents",
        ],
        "problems": [],
    })
    out_path.write_text(json.dumps(v2, indent=1, ensure_ascii=False))
    print(f"wrote {out_path}: {len(changelog)} boundary changes, hedged={len(hedged)} {hedged},"
          f" target_calc={sorted(C3_TARGET_CALC)}, clean cuts {clean}/{total}")
    for c in changelog:
        print(" -", c)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
