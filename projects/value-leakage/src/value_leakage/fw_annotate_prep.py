"""Prepare cut-annotation worksheets for a fw_parents run (protocol v3, Gate C).

Mechanical only — no cut is chosen here. For each parent this script:
  - splits `reasoning` into sentences (offsets preserved);
  - lists CANDIDATE c2 sentences (population-like: a number 50k-200k near a
    giraffe/population word) and CANDIDATE c3 sentences (spots-like: a number
    30-5000 near a spot word), each with surrounding context;
  - computes the c4 cut = token index of `</think>` (reasoning only, excludes
    `</think>`, the answer, and the end-turn token);
  - loads the echo token/text map and provides char->token boundary lookup for
    the completion region.

The human/agent annotator then chooses c2/c3 sentence ids (or marks the cell
missing) and E3 eligibility; `--emit` writes the frozen manifest with token
indices, decoded-prefix tails, and rule-compliance checks.

  uv run python -m value_leakage.fw_annotate_prep --run_dir=runs/fw_parents_<stamp> worksheet
  uv run python -m value_leakage.fw_annotate_prep --run_dir=... emit --decisions=<file>
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

SENT_RE = re.compile(r".+?(?:[.!?](?=[\s\"')\]]|$)|\n)+|.+$", re.S)
POP_WORD = re.compile(r"giraffe|population|individu|alive|wild|captiv", re.I)
POP_NUM = re.compile(r"(?<![\w.,])(\d{2,3},?\d{3})(?![\w])")
SPOT_WORD = re.compile(r"spot|patch", re.I)
SPOT_NUM = re.compile(r"(?<![\w.,])(\d{2,4})(?![\w,])")


def sentences(text: str) -> list[dict]:
    out, pos = [], 0
    for m in SENT_RE.finditer(text):
        s = m.group(0)
        out.append({"sid": len(out), "start": m.start(), "end": m.end(),
                    "text": s.strip()})
        pos = m.end()
    assert pos >= len(text.rstrip()) - 1 if text else True
    return out


def load_parent(run_dir: Path, parent_id: str) -> dict:
    p = json.loads((run_dir / f"parent_{parent_id}.json").read_text())["response"]
    m = json.loads((run_dir / f"map_{parent_id}.json").read_text())["response"]
    choice = p["choices"][0]
    raw = choice["raw_output"]
    lp = m["choices"][0]["logprobs"]
    n_prompt = len(raw["prompt_token_ids"])
    echoed = m["choices"][0]["text"]
    # completion-region token/text map (exclude the 1 generated map token)
    n_total = n_prompt + len(raw["completion_token_ids"])
    offs = lp["text_offset"][:n_total]
    toks = lp["tokens"][:n_total]
    comp_start_char = offs[n_prompt]
    return {
        "parent_id": parent_id,
        "reasoning": choice["message"].get("reasoning_content") or "",
        "content": choice["message"].get("content") or "",
        "completion_ids": raw["completion_token_ids"],
        "prompt_ids": raw["prompt_token_ids"],
        "echoed_text": echoed,
        "comp_token_offsets": [o - comp_start_char for o in offs[n_prompt:]],
        "comp_tokens": toks[n_prompt:],
        "completion_text": echoed[comp_start_char:],
    }


def char_to_token_boundary(par: dict, char_end: int) -> dict:
    """Smallest completion-token boundary whose decoded prefix covers char_end.
    Returns the boundary, the decoded tail, and whether only whitespace was
    added beyond char_end (the frozen rule; anything else -> not clean)."""
    offsets = par["comp_token_offsets"]
    for j, off in enumerate(offsets):
        if off >= char_end:
            added = par["completion_text"][char_end:off]
            return {"token_index": j, "boundary_char": off,
                    "added_beyond_cut": added, "clean": added.strip() == ""}
    return {"token_index": None, "boundary_char": None, "added_beyond_cut": None, "clean": False}


def worksheet(run_dir: str) -> None:
    run = Path(run_dir)
    manifest = json.loads((run / "manifest.json").read_text())
    sheets = []
    for row in manifest:
        if row.get("status") != "ok":
            sheets.append({"parent_id": row["parent_id"], "status": row.get("status")})
            continue
        par = load_parent(run, row["parent_id"])
        sents = sentences(par["reasoning"])
        cand_pop = [s for s in sents
                    if POP_WORD.search(s["text"]) and POP_NUM.search(s["text"])]
        cand_spot = [s for s in sents
                     if SPOT_WORD.search(s["text"]) and SPOT_NUM.search(s["text"])]
        sheets.append({
            "parent_id": row["parent_id"],
            "n_sentences": len(sents),
            "reasoning_chars": len(par["reasoning"]),
            "final_line": row.get("final_line"),
            "candidate_c2": [{"sid": s["sid"], "end": s["end"], "text": s["text"][:220]}
                             for s in cand_pop[:12]],
            "candidate_c3": [{"sid": s["sid"], "end": s["end"], "text": s["text"][:220]}
                             for s in cand_spot[:20]],
        })
    (run / "annotation_worksheet.json").write_text(json.dumps(sheets, indent=1, ensure_ascii=False))
    print(f"wrote {run/'annotation_worksheet.json'} ({len(sheets)} parents)")


def emit(run_dir: str, decisions: str) -> None:
    """decisions: JSON list of {parent_id, c2_sid|null, c3_sid|null,
    c2_missing_reason?, c3_missing_reason?, e3_sid|null, e3_note?, notes?}.
    Emits parent_annotations.json with frozen token boundaries and checks."""
    run = Path(run_dir)
    manifest = {r["parent_id"]: r for r in json.loads((run / "manifest.json").read_text())}
    decs = json.loads(Path(decisions).read_text())
    out_rows, problems = [], []
    for d in decs:
        pid = d["parent_id"]
        row = manifest[pid]
        par = load_parent(run, pid)
        sents = sentences(par["reasoning"])
        rec = {"parent_id": pid, "condition": row["condition"],
               "natural_stop": row["natural_stop"],
               "final_line": row.get("final_line"),
               "incentive_first_mention_char": row.get("incentive_first_mention_char"),
               "decisions": d}
        # c1 deterministic; c4 = token index of </think> in completion ids
        cuts = {}
        c1_end = sents[0]["end"] if sents else None
        for name, char_end in (("c1", c1_end),
                               ("c2", sents[d["c2_sid"]]["end"] if d.get("c2_sid") is not None else None),
                               ("c3", sents[d["c3_sid"]]["end"] if d.get("c3_sid") is not None else None)):
            if char_end is None:
                cuts[name] = {"missing": True,
                              "reason": d.get(f"{name}_missing_reason", "no qualifying sentence")}
                continue
            b = char_to_token_boundary(par, char_end)
            sent = next(s for s in sents if s["end"] == char_end)
            cuts[name] = {"missing": False, "char_end": char_end,
                          "sentence_text": sent["text"],
                          "prefix_tail_decoded": par["completion_text"][:b["boundary_char"]][-80:] if b["token_index"] is not None else None,
                          "incentive_visible": (row.get("incentive_first_mention_char") is not None
                                                and row["incentive_first_mention_char"] < char_end),
                          **b}
            if not b["clean"]:
                problems.append(f"{pid}:{name} cut not on clean token boundary (adds {b['added_beyond_cut']!r})")
        try:
            c4_tok = par["completion_ids"].index(154842)  # </think>
            cuts["c4"] = {"missing": False, "token_index": c4_tok,
                          "excludes": ["</think>", "answer", "end-of-turn"],
                          "incentive_visible": row.get("incentive_first_mention_char") is not None,
                          "clean": True}
        except ValueError:
            cuts["c4"] = {"missing": True, "reason": "no </think> token (length-capped)"}
        # ordering check
        order = [cuts[n].get("token_index") for n in ("c1", "c2", "c3", "c4")
                 if not cuts[n].get("missing")]
        if order != sorted(order) or len(set(order)) != len(order):
            problems.append(f"{pid}: cut order violation {order}")
        rec["cuts"] = cuts
        rec["e3_eligible"] = d.get("e3_sid") is not None
        if d.get("e3_sid") is not None:
            s = sents[d["e3_sid"]]
            rec["e3_sentence"] = {"sid": s["sid"], "start": s["start"], "end": s["end"],
                                  "text": s["text"], "note": d.get("e3_note")}
        out_rows.append(rec)
    annotations = {
        "annotation_status": "frozen_before_any_forced_outcome",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run),
        "rules": "protocol v3 E2/E3; c1 first substantive sentence, c2 first total-population "
                 "selection after c1, c3 first point-valued whole-giraffe spots after c2, "
                 "c4 = token index of </think>; missing cells stay missing; no parent replacement",
        "problems": problems,
        "parents": out_rows,
    }
    path = run / "parent_annotations.json"
    if path.exists():
        raise SystemExit(f"{path} already exists; refusing to overwrite a frozen manifest")
    path.write_text(json.dumps(annotations, indent=1, ensure_ascii=False))
    print(f"wrote {path}; problems: {len(problems)}")
    for p in problems:
        print(" !", p)


if __name__ == "__main__":
    import fire

    fire.Fire({"worksheet": worksheet, "emit": emit})
