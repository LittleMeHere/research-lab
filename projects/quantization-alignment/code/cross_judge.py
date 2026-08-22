#!/usr/bin/env python3
"""Compare Sonnet and Opus labels for FP16-to-NF4 TruthfulQA differences.

The script reports per-response agreement, each judge's paired difference, and a
descriptive difference on the subset where both judges agree. Because that subset
is selected using agreement, it is not an independent replication.

Reads judge_capability_results.json (Sonnet) + judge_capability_results_opus.json.

Usage:
    python code/cross_judge.py
"""

import json
import os
import sys

import numpy as np

import capability_analysis as C   # reuse paired_delta, judged, short, etc.

sys.stdout.reconfigure(encoding="utf-8")

SON_FILE = C.CAP_FILE
OPUS_FILE = os.path.join(C.DATA_DIR, "judge_capability_results_opus.json")


def cells(file):
    return C.judged(file, "correct", bool)   # model -> {quant -> {idx: 0/1}}


def agreement(s, o):
    """Per-judge agreement over all shared (quant, idx) for one model."""
    same = tot = 0
    for q in s:
        if q in o:
            for idx in set(s[q]) & set(o[q]):
                tot += 1
                same += int(s[q][idx] == o[q][idx])
    return (same / tot * 100, tot) if tot else (None, 0)


def consensus_delta(s, o):
    """FP16-to-NF4 difference on responses for which both judges give the same label."""
    def agreed(q):
        if q not in s or q not in o:
            return {}
        return {i: s[q][i] for i in set(s[q]) & set(o[q]) if s[q][i] == o[q][i]}
    return C.paired_delta(agreed("fp16"), agreed("nf4_dq"))


def main():
    if not os.path.exists(OPUS_FILE):
        print("Run `python judge_rescore.py capability --backend opus` first."); return
    son, opu = cells(SON_FILE), cells(OPUS_FILE)

    print("=" * 92)
    print("  SONNET/OPUS COMPARISON — TruthfulQA under NF4 (fp16 -> nf4_dq)")
    print("  Sonnet Δ | Opus Δ | label agreement | both-agree subset Δ")
    print("=" * 92)
    print(f"  {'model':<22} {'Sonnet Δ':>20} {'Opus Δ':>20} {'agree':>7} {'consensus Δ':>20}")
    print("-" * 92)

    for mid in sorted(son, key=C.short):
        s, o = son.get(mid, {}), opu.get(mid, {})
        if not all(q in s for q in ("fp16", "nf4_dq")) or not all(q in o for q in ("fp16", "nf4_dq")):
            continue
        sd = C.paired_delta(s["fp16"], s["nf4_dq"])
        od = C.paired_delta(o["fp16"], o["nf4_dq"])
        cd = consensus_delta(s, o)
        ag, _ = agreement(s, o)
        agtxt = f"{ag:.0f}%" if ag is not None else "n/a"
        print(f"  {C.short(mid):<22} {C.fmt(sd):>20} {C.fmt(od):>20} {agtxt:>7} {C.fmt(cd):>20}")

    print("-" * 92)
    print("  Sonnet and Opus are different models from the same provider.")
    print("  The both-agree subset is selected on agreement and is not an independent replication.")
    print("=" * 92)


if __name__ == "__main__":
    main()
