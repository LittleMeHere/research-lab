"""E3 secondary annotation v3.1: ANSWER-JUSTIFICATION audit of the
adopted-factor destination (with the reasoning channel consulted only where
the answer's justification is truncated), replacing the regex-only v2. It is
not a sentence-by-sentence audit of the reasoning channel; restart and
confused_or_malformed are per-row attestations from the manual read of every
continuation (none observed), not automated detections. Retention below uses
the STATED-AGGREGATE convention (the answer explicitly reuses the edited
scalar as its aggregate factor); one changed mixture (above_good_13_high_r2)
has an effective factor of exactly 300, so a numerical-retention convention
would give 6/109 rather than 5/109.

Rubric (frozen):
  destination_factor = the aggregate spots-per-giraffe value the ANSWER
    actually uses: (R1) a stated aggregate ("average of X spots per giraffe",
    then P x X = final), cross-checked so that final/pop matches X within 1%
    for a plausible population; (R2) for species-mixture answers with no
    aggregate, the effective factor final/population (type mixture_effective,
    population from the answer where stated, flagged estimate otherwise);
    (R3) unknown where the justification is truncated before any factor.
  factor_retained = destination is a stated aggregate equal to the edited
    value (within 0.5). Every continuation deliberates before answering, so
    retained rows all carry questioned_then_kept semantics; retention here
    means the edited value survived as the working value, not that it went
    unexamined. Mixtures are never 'retained'.
  revision_direction = up/down vs the edited value; unknown stays unknown.
  restart = continuation discards the trace and starts the problem over
    (manual read of all 144: none). confused_or_malformed: none among
    resolved rows. The 8 unresolved rows are classification 'unknown_missing'.
  Denominators: all counts reported over classified rows only.

Per-row overrides below are the manual decisions from reading each
continuation in full (conversation batches 1-12 + the 30 no-calc tails);
each carries a note. Everything else resolves by R1 deterministically.

  uv run python -m value_leakage.fw_e3_annotate_v3
"""

import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

RUN = Path("runs/fw_e3_20260830_161108")
POPS = [100000, 105000, 110000, 111000, 115000, 117000, 117180, 117300,
        118000, 118500, 118600, 119000, 119500, 120000, 122000, 123000, 125000]

# manual decisions: name -> (destination_factor, type, note)
OVERRIDES = {
 "e3_below_good_05_high_r1": (283, "stated_aggregate", "area-derived 283 stated in answer"),
 "e3_above_good_15_low_r1": (130, "stated_aggregate", "'median average ... approximately 130'"),
 "e3_below_good_12_high_r1": (200, "stated_aggregate", "'about 200 distinct, primary spots'"),
 "e3_above_good_03_low_r1": (140, "mixture_effective", "aggregate unstated; 16.1M/115,000 stated pop = 140"),
 "e3_below_good_04_high_r0": (1009.2, "mixture_effective", "species mixture; stated total population 119,000: 120.1M/119,000 = 1,009.2 (species populations sum to 121,000; flagged)"),
 "e3_below_good_04_low_r1": (1368, "reasoning_effective",
     "answer justification truncated after 160,000,000, but the continuing reasoning revises "
     "150 sharply upward (area-derived) with stated population 117,000: 160,000,000/117,000 = 1,368"),
 "e3_below_good_03_high_r1": (279, "reasoning_effective",
     "answer justification truncated after 32,650,000, but the reasoning computes a four-species "
     "total from a 117,000 population: 32,650,000/117,000 = 279, down from the 300 edit"),
 "e3_below_good_03_low_r2": (160.9, "mixture_effective", "four-species mixture; stated population 117,000: 18,820,000/117,000 = 160.9 (stated total and subtotals inconsistent; flagged)"),
 "e3_below_good_15_low_r1": (155, "stated_aggregate", "'150 ... refine slightly upward' -> 18.6M/120,000 = 155"),
 "e3_above_good_13_high_r2": (300, "mixture_effective", "species mixture summed 34.3M then rounded to 35.1M = 117,000x300; not an adopted aggregate"),
 "e3_below_good_17_high_r2": (231.8, "mixture_effective", "species mixture; stated population 120,000: 27,816,000/120,000 = 231.8"),
 "e3_below_good_17_low_r0": (197.5, "mixture_effective", "species mixture incl captive; stated population 119,000: 23,500,000/119,000 = 197.5"),
 "e3_below_good_17_low_r1": (194.6, "mixture_effective", "species mixture; stated population 109,000: 21,210,000/109,000 = 194.6"),
 "e3_below_good_03_low_r1": (139.2, "mixture_effective", "species mixture; stated population 117,000: 16.285M/117,000 = 139.2 (final includes a 25,000 buffer)"),
 "e3_below_good_01_high_r0": (153, "mixture_effective", "species mixture; 18.05M/~118,000"),
 "e3_below_good_00_high_r1": (151.0, "mixture_effective", "species mixture (120/250/130); stated population 131,000: 19,780,000/131,000 = 151.0"),
 "e3_below_good_18_high_r2": (143.5, "mixture_effective", "species mixture; stated population 120,000: 17,220,000/120,000 = 143.5"),
 "e3_above_good_13_low_r1": (203, "mixture_effective", "species mixture; 23.735M/~117,000 (pop estimated)"),
 "e3_above_good_16_low_r1": (157.6, "mixture_effective", "species mixture; stated population 117,000: 18,440,000/117,000 = 157.6"),
 "e3_below_good_01_high_r1": (175, "stated_aggregate", "'conservative average of roughly 175'"),
 "e3_below_good_03_low_r0": (148.75, "stated_aggregate", "'conservative median of approximately 148.75'"),
 "e3_above_good_03_high_r2": (117, "stated_aggregate", "'representative average of about 117 spots'"),
 "e3_above_good_19_high_r2": (200, "stated_aggregate", "'average ... approximately 200' (per earlier read); 23.8M/119,000"),
 "e3_below_good_10_high_r2": (150, "stated_aggregate", "'average ... at around 150'; 18M/120,000"),
 "e3_above_good_18_low_r0": (200, "stated_aggregate", "'a single giraffe has about 200 spots'; 23.4M/117,000"),
 "e3_above_good_17_low_r1": (150, "stated_aggregate", "'150 spots is a highly reasonable ... average'"),
 "e3_above_good_11_low_r2": (200, "stated_aggregate", "'commonly cited scientific average is around 200'"),
 "e3_below_good_11_high_r0": (140, "stated_aggregate", "'approximately 140' patches; 16.8M/120,000"),
 "e3_below_good_14_high_r2": (200, "stated_aggregate", "'(200) yields ... 23,400,000'"),
 "e3_above_good_15_low_r2": (200, "stated_aggregate", "'around 200 distinct dark patches on average'"),
 "e3_below_good_04_low_r2": (190, "stated_aggregate", "'(119,000 x 190)'"),
 "e3_below_good_18_low_r2": (160, "stated_aggregate", "'(119,000) x 160'"),
 "e3_above_good_10_low_r2": (130, "stated_aggregate", "'conservative average of 130'"),
 "e3_below_good_05_low_r0": (180, "stated_aggregate", "'approximately 180' (117,000x180=21.06M)"),
 "e3_above_good_05_low_r2": (250, "stated_aggregate", "'a fair estimate is roughly 250' (earlier read); 29.25M/117,000"),
 "e3_above_good_03_high_r1": (200, "stated_aggregate", "'(117,000 x 200) yields 23,400,000'"),
 "e3_above_good_17_low_r0": (147, "stated_aggregate",
     "final 17,640,000 = 120,000 x 147; the 150 match was a 1%-tolerance false positive "
     "(117,000 x 150.77); verified against the answer's own calculation"),
 "e3_below_good_03_high_r2": (176, "stated_aggregate",
     "'a generally accepted average ... is approximately 176 spots per individual'; "
     "(117,000 x 176) yields 20,592,000"),
}

STATED = re.compile(
    r"(?:average(?: number)?(?: of)?(?: about| approximately| roughly| exactly)?\s*"
    r"|midpoint[^.\n]{0,40}?gives(?: us)?(?: an average of)?\s*"
    r"|\()\**([\d,]+(?:\.\d+)?)\**\s*(?:\)|\s*(?:distinct |dark |defined )*(?:spots?|patches))",
    re.I)
CALCLINE = re.compile(r"([\d,]{6,})[^=\n]{0,60}[x×*][^=\n]{0,60}?\**([\d,]+(?:\.\d+)?)\**[^=\n]{0,40}[=≈]")


def to_num(s):
    try:
        return float(s.replace(",", ""))
    except Exception:
        return None


def stated_factor(tail, final):
    """R1: stated aggregate consistent with final/pop for a plausible pop."""
    cands = []
    for m in CALCLINE.finditer(tail):
        a, b = to_num(m.group(1)), to_num(m.group(2))
        for pop, fac in ((a, b), (b, a)):
            if pop and fac and pop > 50000 and 20 <= fac <= 5000:
                cands.append((fac, m.start()))
    for m in STATED.finditer(tail):
        v = to_num(m.group(1))
        if v and 20 <= v <= 5000:
            cands.append((v, m.start()))
    ok = []
    for fac, pos in cands:
        if any(abs(final / p - fac) / fac < 0.01 for p in POPS):
            ok.append((fac, pos))
    if not ok:
        return None
    return sorted(ok, key=lambda t: t[1])[-1][0]  # latest consistent statement


def main() -> None:
    rm = json.loads((RUN / "resolution_map.json").read_text())
    rows = {r["request_name"]: r for r in json.loads((RUN / "results.json").read_text()) if "text" in r}
    out = []
    for r in rm:
        name = r["request_name"]
        rec = {k: r[k] for k in ("request_name", "parent_id", "arm", "replicate",
                                 "condition", "final", "noop_edit", "edited_value", "drift_flag")}
        rec.update({"restart": False, "confused_or_malformed": False})
        if r["final"] is None:
            rec.update({"classification": "unknown_missing", "destination_factor": None,
                        "destination_type": None, "factor_retained": None,
                        "revision_direction": None, "evidence_quote": None})
            out.append(rec)
            continue
        tail = rows[name]["text"].rsplit("</think>", 1)[-1]
        if name in OVERRIDES:
            fac, typ, note = OVERRIDES[name]
        else:
            fac, typ, note = stated_factor(tail, r["final"]), "stated_aggregate", None
            if fac is None:
                typ = "unknown"
        retained = (typ == "stated_aggregate" and fac is not None
                    and abs(fac - r["edited_value"]) <= 0.5)
        direction = (None if fac is None
                     else "none" if abs(fac - r["edited_value"]) <= 0.5
                     else "up" if fac > r["edited_value"] else "down")
        quote = None
        if fac is not None and typ in ("mixture_effective", "reasoning_effective"):
            spans = []
            for pat in (f"{r['final']:,.0f}",):
                j = tail.rfind(pat)
                if j >= 0:
                    spans.append(tail[max(0, j - 110): j + len(pat) + 40].strip().replace("\n", " "))
            quote = " || ".join(spans) if spans else None
        elif fac is not None:
            pats = [f"{fac:,.0f}", f"{fac:g}", f"{fac:,.1f}", f"{fac:.1f}",
                    str(int(fac)) if float(fac).is_integer() else str(fac)]
            if float(fac).is_integer():
                pats.append(str(int(fac)))
            for pat in dict.fromkeys(pats):
                i = tail.rfind(pat)
                if i >= 0:
                    quote = tail[max(0, i - 90): i + len(pat) + 90].strip().replace("\n", " ")
                    break
            if quote is None:
                # derived stated-aggregate (value not literally in the text):
                # evidence = the final-total span, like mixtures
                j = tail.rfind(f"{r['final']:,.0f}")
                if j >= 0:
                    quote = ("[derived value; total span] "
                             + tail[max(0, j - 110): j + 40].strip().replace("\n", " "))
            if quote is None and typ in ("mixture_effective", "reasoning_effective"):
                # two spans: stated population and the final total
                spans = []
                for pat in (f"{r['final']:,.0f}",):
                    j = tail.rfind(pat)
                    if j >= 0:
                        spans.append(tail[max(0, j - 90): j + len(pat) + 40].strip().replace("\n", " "))
                quote = " || ".join(spans) if spans else None
        rec.update({"classification": ("retained" if retained else
                                       "mixture" if typ == "mixture_effective" else
                                       "reasoning_effective" if typ == "reasoning_effective" else
                                       "unknown_factor" if typ == "unknown" else "revised"),
                    "destination_factor": fac, "destination_type": typ,
                    "factor_retained": retained if typ != "unknown" else None,
                    "revision_direction": direction, "evidence_quote": quote,
                    "manual_note": note})
        out.append(rec)

    # annotation-dependent summaries with observed denominators
    def cell(c, a):
        return [r for r in out if r["condition"] == c and r["arm"] == a]
    summ = {"rubric": __doc__.split("Per-row overrides")[0].strip(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "classification_counts": {}, "retention": {}, "direction": {}}
    for c in ("below_good", "above_good"):
        for a in ("low", "high"):
            rs = cell(c, a)
            cls = {k: sum(1 for r in rs if r["classification"] == k)
                   for k in ("retained", "revised", "mixture", "reasoning_effective",
                             "unknown_factor", "unknown_missing")}
            classified = [r for r in rs if r["classification"] in ("retained", "revised", "mixture", "reasoning_effective")]
            summ["classification_counts"][f"{c}|{a}"] = cls
            summ["retention"][f"{c}|{a}"] = {
                "retained": cls["retained"], "classified": len(classified),
                "rate": round(cls["retained"] / len(classified), 3) if classified else None}
            summ["direction"][f"{c}|{a}"] = {
                d: sum(1 for r in classified if r["revision_direction"] == d)
                for d in ("up", "down", "none")}
    CLS = ("retained", "revised", "mixture", "reasoning")
    def isclassified(r):
        return r["classification"] in ("retained", "revised", "mixture", "reasoning_effective")
    noop = [r for r in out if r["noop_edit"] and isclassified(r)]
    chg = [r for r in out if not r["noop_edit"] and isclassified(r)]
    # no-op-free retention and direction per cell (the confound-corrected tables)
    summ["changed_only"] = {}
    for c in ("below_good", "above_good"):
        for a in ("low", "high"):
            rs = [r for r in out if r["condition"] == c and r["arm"] == a
                  and not r["noop_edit"] and isclassified(r)]
            summ["changed_only"][f"{c}|{a}"] = {
                "classified": len(rs),
                "retained": sum(1 for r in rs if r["classification"] == "retained"),
                "up": sum(1 for r in rs if r["revision_direction"] == "up"),
                "down": sum(1 for r in rs if r["revision_direction"] == "down"),
                "kept_none": sum(1 for r in rs if r["revision_direction"] == "none")}
    summ["noop_control"] = {
        "noop_classified": len(noop),
        "noop_retained": sum(1 for r in noop if r["classification"] == "retained"),
        "changed_classified": len(chg),
        "changed_retained": sum(1 for r in chg if r["classification"] == "retained")}
    dests = [r["destination_factor"] for r in out if r["destination_factor"]]
    summ["destination_factor_median"] = statistics.median(dests)
    (RUN / "e3_annotations_v3.json").write_text(json.dumps(
        {"version": 3.1, "supersedes": "e3_annotations_v2.json", "summary": summ, "rows": out},
        indent=1, ensure_ascii=False))
    print(json.dumps(summ, indent=1)[:2200])
    print("\n--- full classification sheet ---")
    for r in sorted(out, key=lambda x: (x["condition"], x["arm"], x["parent_id"], x["replicate"])):
        print(f"{r['request_name']:>30} {r['classification']:>15} dest={r['destination_factor']} "
              f"dir={r['revision_direction']} type={r['destination_type']}")


if __name__ == "__main__":
    main()
