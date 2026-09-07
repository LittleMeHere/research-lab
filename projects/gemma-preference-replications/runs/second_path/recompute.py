#!/usr/bin/env python3
"""Blind recomputation of the estimands in BRIEF.md from raw/ only."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"


def read_json(name):
    with (RAW / name).open(encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(name):
    with (RAW / name).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_csv(name):
    with (RAW / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def p_a(row):
    """P(A) renormalized over the A/B answer tokens, computed stably."""
    d = float(row["logp_a"]) - float(row["logp_b"])
    if d >= 0:
        return 1.0 / (1.0 + math.exp(-d))
    e = math.exp(d)
    return e / (1.0 + e)


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return None
    xc, yc = x - x.mean(), y - y.mean()
    den = math.sqrt(float(xc @ xc) * float(yc @ yc))
    return float((xc @ yc) / den) if den else None


def wilson(successes, n, z=1.959963984540054):
    if n == 0:
        return [None, None]
    phat = successes / n
    z2 = z * z
    center = (phat + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt(phat * (1 - phat) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return [center - half, center + half]


def fisher_ci(r, n, zcrit=1.959963984540054):
    if n <= 3 or r is None:
        return [None, None]
    # Guard only against roundoff at a mathematically perfect correlation.
    rr = min(max(r, -1 + 1e-15), 1 - 1e-15)
    z = math.atanh(rr)
    half = zcrit / math.sqrt(n - 3)
    return [math.tanh(z - half), math.tanh(z + half)]


def sigmoid_array(x):
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def lbfgs(fun_grad, x0, max_iter=500, memory=12, gtol=1e-8):
    """Small, deterministic L-BFGS minimizer with Armijo backtracking."""
    x = np.asarray(x0, dtype=float).copy()
    f, g = fun_grad(x)
    s_hist, y_hist, rho_hist = [], [], []
    converged = False

    for iteration in range(max_iter):
        if float(np.max(np.abs(g))) <= gtol:
            converged = True
            break

        q = g.copy()
        alphas = []
        for s, yv, rho in zip(reversed(s_hist), reversed(y_hist), reversed(rho_hist)):
            alpha = rho * float(s @ q)
            alphas.append(alpha)
            q -= alpha * yv
        if s_hist:
            sy = float(s_hist[-1] @ y_hist[-1])
            yy = float(y_hist[-1] @ y_hist[-1])
            gamma = sy / yy if yy > 0 else 1.0
        else:
            gamma = 1.0
        r = gamma * q
        for s, yv, rho, alpha in zip(s_hist, y_hist, rho_hist, reversed(alphas)):
            r += s * (alpha - rho * float(yv @ r))
        direction = -r
        gd = float(g @ direction)
        if not math.isfinite(gd) or gd >= 0:
            direction = -g
            gd = -float(g @ g)
            s_hist, y_hist, rho_hist = [], [], []

        step = 1.0
        while True:
            xn = x + step * direction
            fn, gn = fun_grad(xn)
            if math.isfinite(fn) and fn <= f + 1e-4 * step * gd:
                break
            step *= 0.5
            if step < 1e-16:
                raise RuntimeError("L-BFGS line search failed")

        s = xn - x
        yv = gn - g
        sy = float(s @ yv)
        if sy > 1e-14:
            if len(s_hist) == memory:
                s_hist.pop(0); y_hist.pop(0); rho_hist.pop(0)
            s_hist.append(s); y_hist.append(yv); rho_hist.append(1.0 / sy)
        x, f, g = xn, fn, gn
    else:
        iteration = max_iter

    return x, {
        "converged": converged,
        "iterations": iteration,
        "objective": float(f),
        "max_abs_gradient": float(np.max(np.abs(g))),
    }


def pairwise_estimands(rows):
    ps = [p_a(r) for r in rows]
    by_pair = defaultdict(dict)
    for r, p in zip(rows, ps):
        by_pair[r["pair_id"]][r["order"]] = p
    paired = [v for v in by_pair.values() if 0 in v and 1 in v]
    consistent = sum((v[0] > 0.5) == (v[1] < 0.5) for v in paired)
    return {
        "n_rows": len(rows),
        "mean_mass": mean([float(r["mass"]) for r in rows]),
        "fraction_mass_lt_0_5": mean([float(r["mass"]) < 0.5 for r in rows]),
        "slot_A_bias_mean_renormalized_p_A": mean(ps),
        "order_agreement_fraction": consistent / len(paired),
        "order_agreement_n_pairs": len(paired),
        "order_agreement_n": consistent,
    }


def fit_bradley_terry(rows, pool, supplied):
    eligible = [r for r in rows if float(r["mass"]) >= 0.5]
    ids = [r["id"] for r in pool]
    index = {task_id: i for i, task_id in enumerate(ids)}
    a = np.asarray([index[r["task_a"]] for r in eligible], dtype=np.int64)
    b = np.asarray([index[r["task_b"]] for r in eligible], dtype=np.int64)
    y = np.asarray([p_a(r) for r in eligible], dtype=float)
    n_rows, n_tasks = len(eligible), len(ids)
    penalty = 0.01

    def fun_grad(theta):
        u, beta = theta[:-1], theta[-1]
        eta = u[a] - u[b] + beta
        loss = float(np.mean(np.logaddexp(0.0, eta) - y * eta) + penalty * np.mean(u * u))
        residual = (sigmoid_array(eta) - y) / n_rows
        grad_u = np.bincount(a, weights=residual, minlength=n_tasks)
        grad_u -= np.bincount(b, weights=residual, minlength=n_tasks)
        grad_u += (2 * penalty / n_tasks) * u
        grad = np.empty(n_tasks + 1)
        grad[:-1] = grad_u
        grad[-1] = residual.sum()
        return loss, grad

    theta, diagnostics = lbfgs(fun_grad, np.zeros(n_tasks + 1))
    u, beta = theta[:-1], float(theta[-1])
    pool_by_id = {r["id"]: r for r in pool}
    origin_values = defaultdict(list)
    for task_id, value in zip(ids, u):
        origin_values[pool_by_id[task_id]["origin"]].append(float(value))
    supplied_u = {r["id"]: float(r["u"]) for r in supplied}
    common = [i for i in ids if i in supplied_u]
    corr = pearson([u[index[i]] for i in common], [supplied_u[i] for i in common])
    return {
        "n_rows_mass_ge_0_5": n_rows,
        "n_tasks": n_tasks,
        "beta_slot_A": beta,
        "mean_u_by_origin": {k: mean(v) for k, v in sorted(origin_values.items())},
        "pearson_r_with_utilities_csv": corr,
        "correlation_n_tasks": len(common),
        "l2_penalty": "0.01 * mean(u^2)",
        "fit_diagnostics": diagnostics,
    }


def effective_mode(row):
    return row.get("mode", "contrastive")


def steering_swing(rows, direction, layer, c_abs):
    cells = defaultdict(dict)
    for r in rows:
        if (effective_mode(r) == "contrastive" and r["dir"] == direction
                and int(r["layer"]) == layer and abs(float(r["c"])) == c_abs
                and float(r["c"]) != 0):
            cells[(r["pair_id"], r["order"])][1 if float(r["c"]) > 0 else -1] = p_a(r)
    paired = [v for v in cells.values() if 1 in v and -1 in v]
    swings = [v[1] - v[-1] for v in paired]
    return {"E": mean(swings), "n_trials": len(swings)}


def steering_estimands(rows):
    probe_rows = [r for r in rows if effective_mode(r) == "contrastive" and r["dir"] == "probe"]
    layers_005 = sorted({int(r["layer"]) for r in probe_rows if abs(float(r["c"])) == 0.05})
    cs_l23 = sorted({abs(float(r["c"])) for r in probe_rows
                     if int(r["layer"]) == 23 and float(r["c"]) != 0})
    table_005 = {str(layer): steering_swing(rows, "probe", layer, 0.05) for layer in layers_005}
    table_l23 = {format(c, "g"): steering_swing(rows, "probe", 23, c) for c in cs_l23}

    prefixes = ("rand", "top", "shuf", "cov", "low")
    dirs = sorted({r["dir"] for r in rows if effective_mode(r) == "contrastive"
                   and int(r["layer"]) == 23 and abs(float(r["c"])) == 0.05
                   and r["dir"].startswith(prefixes)})
    controls = {d: steering_swing(rows, d, 23, 0.05) for d in dirs}
    family_means = {}
    for prefix in prefixes:
        vals = [abs(v["E"]) for d, v in controls.items() if d.startswith(prefix) and v["E"] is not None]
        family_means[prefix] = {"mean_abs_E": mean(vals), "n_directions": len(vals)}

    gate = [p_a(r) for r in probe_rows if int(r["layer"]) == 23 and float(r["c"]) == 0]
    return {
        "probe_c_0_05_by_layer": table_005,
        "probe_layer_23_by_c": table_l23,
        "controls_layer_23_c_0_05": controls,
        "control_family_means_abs_E": family_means,
    }, {
        "mean_renormalized_p_A": mean(gate),
        "n_trials": len(gate),
        "filter": "effective mode contrastive, dir probe, layer 23, c=0",
    }


def patching_estimands(rows):
    baseline_rows = [r for r in rows if r["cond"] == "none"]
    baseline = {(r["pair_id"], r["order"]): p_a(r) for r in baseline_rows}
    confident = {k for k, p in baseline.items() if abs(p - 0.5) > 0.2}

    def sign(value):
        return 1 if value > 0.5 else (-1 if value < 0.5 else 0)

    def rate(cond, layer, restrict_opposite_order=False):
        candidates = []
        for r in rows:
            if r["cond"] != cond or int(r["layer"]) != layer:
                continue
            key = (r["pair_id"], r["order"])
            if key not in confident:
                continue
            if restrict_opposite_order:
                other = (r["pair_id"], 1 - r["order"])
                if other not in baseline or sign(baseline[other]) != -sign(baseline[key]):
                    continue
            candidates.append(sign(baseline[key]) != sign(p_a(r)))
        return {"flip_rate": mean(candidates), "n_trials": len(candidates), "n_flips": sum(candidates)}

    layers = sorted({int(r["layer"]) for r in rows if r["cond"] in ("swap", "eot")})
    return {
        "baseline": {
            "n_trials": len(baseline),
            "confidence_rule": "abs(P(A)-0.5) > 0.2",
            "n_confident_trials": len(confident),
        },
        "swap_by_layer": {str(l): rate("swap", l) for l in layers},
        "eot_by_layer": {str(l): rate("eot", l) for l in layers},
        "eot_all": rate("eot_all", -1),
        "eot_opposite_slot_baseline_by_layer": {
            str(l): rate("eot", l, restrict_opposite_order=True) for l in layers
        },
    }


def value_leakage_estimands(liking, picks, judged):
    scores = defaultdict(list)
    for r in liking:
        if r["parsed"] is not None:
            scores[r["activity"]].append(float(r["parsed"]))
    activity_mean = {a: mean(v) for a, v in scores.items()}

    judge_by_ix = {r["var_ix"]: r["judge"] for r in judged}
    final = []
    selection_counts = Counter()
    appearance_counts = Counter()
    option_counts = Counter()
    higher_n = higher_success = 0
    refusals = 0
    unresolved = 0
    for r in picks:
        choice = judge_by_ix[r["var_ix"]] if r["var_ix"] in judge_by_ix else r["parsed"]
        if choice == "refusal":
            refusals += 1
            final.append(choice)
            continue
        if choice not in (1, 2):
            unresolved += 1
            final.append(None)
            continue
        final.append(choice)
        option_counts[choice] += 1
        a1, a2 = r["activity_1"], r["activity_2"]
        appearance_counts[a1] += 1
        appearance_counts[a2] += 1
        chosen = a1 if choice == 1 else a2
        selection_counts[chosen] += 1
        if a1 in activity_mean and a2 in activity_mean and activity_mean[a1] != activity_mean[a2]:
            higher_n += 1
            if ((choice == 1 and activity_mean[a1] > activity_mean[a2])
                    or (choice == 2 and activity_mean[a2] > activity_mean[a1])):
                higher_success += 1

    rates = {a: selection_counts[a] / appearance_counts[a]
             for a in activity_mean if appearance_counts[a] > 0}
    common = sorted(rates)
    r_value = pearson([activity_mean[a] for a in common], [rates[a] for a in common])
    decisive = option_counts[1] + option_counts[2]
    return {
        "mean_liking_score_by_activity": {a: activity_mean[a] for a in sorted(activity_mean)},
        "selection_rate_by_activity": {a: rates[a] for a in sorted(rates)},
        "score_selection_pearson_r": r_value,
        "score_selection_pearson_n_activities": len(common),
        "score_selection_fisher_z_95pct_CI": fisher_ci(r_value, len(common)),
        "pick_higher_scored_activity": {
            "probability": higher_success / higher_n,
            "wilson_95pct_CI": wilson(higher_success, higher_n),
            "n_success": higher_success,
            "n_decisive_unequal_score_picks": higher_n,
        },
        "refusal_rate": refusals / len(picks),
        "n_refusals": refusals,
        "n_total_rows": len(picks),
        "fraction_decisive_picks_choosing_option_1": option_counts[1] / decisive,
        "n_option_1": option_counts[1],
        "n_option_2": option_counts[2],
        "n_decisive": decisive,
        "n_unresolved_nonrefusal": unresolved,
        "final_choice_rule": "judge verdict for every var_ix in judged.jsonl; otherwise regex parse",
    }


def duplicate_count(rows, key):
    keys = [key(r) for r in rows]
    return len(keys) - len(set(keys))


def anomaly_checks(pool, pairs, pairwise, steer_pairs, steer, patch, picks, judged, liking, utilities):
    anomalies = []
    checks = {
        "duplicate_pairwise_pair_order_keys": duplicate_count(pairwise, lambda r: (r["pair_id"], r["order"])),
        "duplicate_steering_full_cell_keys": duplicate_count(
            steer, lambda r: (r["pair_id"], r["order"], r["layer"], r["c"], r["dir"], effective_mode(r))),
        "duplicate_patching_full_cell_keys": duplicate_count(
            patch, lambda r: (r["pair_id"], r["order"], r["cond"], r["layer"])),
        "duplicate_pick_var_ix": duplicate_count(picks, lambda r: r["var_ix"]),
        "duplicate_judged_var_ix": duplicate_count(judged, lambda r: r["var_ix"]),
        "duplicate_liking_design_cells": duplicate_count(
            liking, lambda r: (r["activity_ix"], r["prompt_ix"], r["rep"])),
    }
    for label, value in checks.items():
        if value:
            anomalies.append(f"{label}: {value}")

    pool_ids = {r["id"] for r in pool}
    utility_ids = {r["id"] for r in utilities}
    if pool_ids != utility_ids:
        anomalies.append(f"pool/utilities ID mismatch: pool-only={len(pool_ids-utility_ids)}, utilities-only={len(utility_ids-pool_ids)}")
    listed_pairs = {(r["pair_id"], o) for r in pairs for o in (0, 1)}
    result_pairs = {(r["pair_id"], r["order"]) for r in pairwise}
    if listed_pairs != result_pairs:
        anomalies.append(f"pairwise cell mismatch: listed-only={len(listed_pairs-result_pairs)}, result-only={len(result_pairs-listed_pairs)}")
    checks["missing_pairwise_pair_order_cells"] = len(listed_pairs - result_pairs)
    heldout = {r["pair_id"] for r in steer_pairs}
    steer_ids = {r["pair_id"] for r in steer}
    patch_ids = {r["pair_id"] for r in patch}
    if heldout != steer_ids:
        anomalies.append(f"steering pair-ID mismatch: listed-only={len(heldout-steer_ids)}, result-only={len(steer_ids-heldout)}")
    if heldout != patch_ids:
        anomalies.append(f"patch pair-ID mismatch: listed-only={len(heldout-patch_ids)}, result-only={len(patch_ids-heldout)}")

    requested_steer = [
        r for r in steer
        if effective_mode(r) == "contrastive" and float(r["c"]) != 0 and (
            (r["dir"] == "probe" and abs(float(r["c"])) == 0.05)
            or (r["dir"] == "probe" and int(r["layer"]) == 23)
            or (int(r["layer"]) == 23 and abs(float(r["c"])) == 0.05
                and r["dir"].startswith(("rand", "top", "shuf", "cov", "low")))
        )
    ]
    steer_signs = defaultdict(set)
    for r in requested_steer:
        key = (r["dir"], int(r["layer"]), abs(float(r["c"])), r["pair_id"], r["order"])
        steer_signs[key].add(1 if float(r["c"]) > 0 else -1)
    missing_steer_signs = sum(len({-1, 1} - signs) for signs in steer_signs.values())
    checks["missing_plus_or_minus_rows_in_requested_steering_trials"] = missing_steer_signs
    if missing_steer_signs:
        anomalies.append(f"missing +c/-c rows in requested steering trials: {missing_steer_signs}")

    baseline_cells = {(r["pair_id"], r["order"]) for r in patch if r["cond"] == "none"}
    patch_layers = sorted({int(r["layer"]) for r in patch if r["cond"] in ("swap", "eot")})
    expected_patch = {
        (pair_id, order, cond, layer)
        for pair_id, order in baseline_cells
        for cond, layers in (("swap", patch_layers), ("eot", patch_layers), ("eot_all", [-1]))
        for layer in layers
    }
    actual_patch = {(r["pair_id"], r["order"], r["cond"], int(r["layer"]))
                    for r in patch if r["cond"] != "none"}
    checks["missing_requested_patch_cells"] = len(expected_patch - actual_patch)
    if expected_patch != actual_patch:
        anomalies.append(f"patch cell mismatch: expected-only={len(expected_patch-actual_patch)}, result-only={len(actual_patch-expected_patch)}")

    missing_like = sum(r["parsed"] is None for r in liking)
    checks["missing_liking_parses"] = missing_like
    if missing_like:
        anomalies.append(f"liking parsed values missing: {missing_like}")
    regex_unparsed = {r["var_ix"] for r in picks if r["parsed"] is None}
    judged_ids = {r["var_ix"] for r in judged}
    if not regex_unparsed.issubset(judged_ids):
        anomalies.append(f"regex-unparsed picks lacking a judge row: {len(regex_unparsed-judged_ids)}")
    checks["regex_unparsed_picks_lacking_judge"] = len(regex_unparsed - judged_ids)

    audit = [r for r in judged if r["regex"] in (1, 2)]
    audit_disagree = sum(r["judge"] in (1, 2) and r["judge"] != r["regex"] for r in audit)
    audit_refusal = sum(r["judge"] == "refusal" for r in audit)
    if audit_disagree or audit_refusal:
        anomalies.append(
            f"judged audit rows override regex: {audit_disagree} decisive disagreements and {audit_refusal} refusals among {len(audit)} audited parsed rows"
        )

    bad_mass = sum(not (0 <= float(r["mass"]) <= 1) for data in (pairwise, steer, patch) for r in data)
    checks["mass_values_outside_0_1"] = bad_mass
    if bad_mass:
        anomalies.append(f"mass outside [0,1]: {bad_mass}")
    low_pairwise = sum(float(r["mass"]) < 0.5 for r in pairwise)
    if low_pairwise:
        anomalies.append(f"pairwise rows with A/B mass below 0.5 (excluded from BT only): {low_pairwise}")
    ties = {
        "pairwise": sum(float(r["logp_a"]) == float(r["logp_b"]) for r in pairwise),
        "steering": sum(float(r["logp_a"]) == float(r["logp_b"]) for r in steer),
        "patching": sum(float(r["logp_a"]) == float(r["logp_b"]) for r in patch),
    }
    checks["exact_logp_A_B_ties"] = ties
    if any(ties.values()):
        anomalies.append(
            "exact A/B log-probability ties (likely affected by logged precision): "
            + ", ".join(f"{k}={v}" for k, v in ties.items())
        )

    # Large null/control effects are numerically valid but scientifically suspicious.
    control_dirs = sorted({r["dir"] for r in steer if r["dir"].startswith(("rand", "top", "shuf", "cov", "low"))})
    control_effects = {d: steering_swing(steer, d, 23, 0.05)["E"] for d in control_dirs}
    finite_controls = {d: e for d, e in control_effects.items() if e is not None}
    max_dir = max(finite_controls, key=lambda d: abs(finite_controls[d]))
    checks["largest_abs_control_steering_E_L23_c_0_05"] = {
        "direction": max_dir, "E": finite_controls[max_dir]
    }
    if abs(finite_controls[max_dir]) > 0.25:
        anomalies.append(
            f"large control steering swings at L23/c=0.05; largest is {max_dir}, E={finite_controls[max_dir]:.6f}"
        )
    return checks, anomalies


def fmt(x, digits=6):
    return "NA" if x is None else f"{x:.{digits}f}"


def make_markdown(results):
    pw, bt = results["pairwise"], results["bradley_terry"]
    st, gate, patch = results["steering_swing"], results["gate_0_baseline"], results["patching"]
    vl, probe = results["value_leakage"], results["probe"]
    lines = [
        "# Recomputed results", "",
        "All probabilities below are renormalized over the A/B answer tokens where applicable.", "",
        "## Pairwise and Bradley–Terry", "",
        f"Pairwise: n={pw['n_rows']}; mean mass={fmt(pw['mean_mass'])}; mass<0.5={fmt(pw['fraction_mass_lt_0_5'])}; mean P(A)={fmt(pw['slot_A_bias_mean_renormalized_p_A'])}; order agreement={fmt(pw['order_agreement_fraction'])} ({pw['order_agreement_n']}/{pw['order_agreement_n_pairs']}).",
        "",
        f"Bradley–Terry: beta={fmt(bt['beta_slot_A'])}; correlation with supplied utilities={fmt(bt['pearson_r_with_utilities_csv'])} (n={bt['correlation_n_tasks']}). Mean u by origin: " + ", ".join(f"{k}={fmt(v)}" for k, v in bt["mean_u_by_origin"].items()) + ".",
        "",
        "## Steering", "",
        "Probe direction, c=0.05:", "",
        "| Layer | E | n |", "|---:|---:|---:|",
    ]
    for layer, v in st["probe_c_0_05_by_layer"].items():
        lines.append(f"| {layer} | {fmt(v['E'])} | {v['n_trials']} |")
    lines += ["", "Probe direction, layer 23:", "", "| c | E | n |", "|---:|---:|---:|"]
    for c, v in st["probe_layer_23_by_c"].items():
        lines.append(f"| {c} | {fmt(v['E'])} | {v['n_trials']} |")
    lines += ["", "Controls at layer 23, c=0.05:", "", "| Direction | E | n |", "|---|---:|---:|"]
    for d, v in st["controls_layer_23_c_0_05"].items():
        lines.append(f"| {d} | {fmt(v['E'])} | {v['n_trials']} |")
    lines += ["", "Control family means of |E|: " + ", ".join(
        f"{k}={fmt(v['mean_abs_E'])}" for k, v in st["control_family_means_abs_E"].items()) + ".", "",
        f"Gate-0 baseline (probe, L23): mean P(A)={fmt(gate['mean_renormalized_p_A'])}, n={gate['n_trials']}.", "",
        "## Patching", "",
        f"The confidence filter retains {patch['baseline']['n_confident_trials']} of {patch['baseline']['n_trials']} baseline trials.", "",
        "| Layer | Swap flip rate | EOT flip rate | EOT restricted flip rate | n | restricted n |", "|---:|---:|---:|---:|---:|---:|",
    ]
    for layer in patch["swap_by_layer"]:
        sw, eo, er = patch["swap_by_layer"][layer], patch["eot_by_layer"][layer], patch["eot_opposite_slot_baseline_by_layer"][layer]
        lines.append(f"| {layer} | {fmt(sw['flip_rate'])} | {fmt(eo['flip_rate'])} | {fmt(er['flip_rate'])} | {eo['n_trials']} | {er['n_trials']} |")
    ea = patch["eot_all"]
    lines += ["", f"EOT-all flip rate: {fmt(ea['flip_rate'])} ({ea['n_flips']}/{ea['n_trials']}).", "",
        "## Value leakage", "",
        f"Across {vl['score_selection_pearson_n_activities']} activities, score/selection Pearson r={fmt(vl['score_selection_pearson_r'])}, Fisher-z 95% CI [{fmt(vl['score_selection_fisher_z_95pct_CI'][0])}, {fmt(vl['score_selection_fisher_z_95pct_CI'][1])}].",
        "",
        f"Higher-scored activity picked with probability {fmt(vl['pick_higher_scored_activity']['probability'])} ({vl['pick_higher_scored_activity']['n_success']}/{vl['pick_higher_scored_activity']['n_decisive_unequal_score_picks']}), Wilson 95% CI [{fmt(vl['pick_higher_scored_activity']['wilson_95pct_CI'][0])}, {fmt(vl['pick_higher_scored_activity']['wilson_95pct_CI'][1])}]. Refusal rate={fmt(vl['refusal_rate'])} ({vl['n_refusals']}/{vl['n_total_rows']}); option-(1) fraction among decisive picks={fmt(vl['fraction_decisive_picks_choosing_option_1'])} ({vl['n_option_1']}/{vl['n_decisive']}).",
        "",
        "Per-activity mean scores and selection rates are in `recompute.json`.", "",
        "## Probe", "",
        f"Maximum held-out r_eval is at layer {probe['layer_of_max_r_eval']} (r_eval={fmt(probe['max_r_eval'])}).", "",
        "## Anomalies", "",
    ]
    anomalies = results["anomalies"]
    if anomalies:
        lines.extend(f"- {a}" for a in anomalies)
    else:
        lines.append("No duplicate keys, missing requested cells, invalid masses, or coverage mismatches were found.")
    lines.append("")
    return "\n".join(lines)


def main():
    pool = read_json("pool.json")
    pairs = read_json("pairwise_pairs.json")
    pairwise = read_jsonl("pairwise_results.jsonl")
    utilities = read_csv("utilities.csv")
    steer_pairs = read_json("steer_pairs.json")
    steer = read_jsonl("steer_results.jsonl")
    patch = read_jsonl("patch_results.jsonl")
    picks = read_jsonl("pick.jsonl")
    judged = read_jsonl("judged.jsonl")
    liking = read_jsonl("liking.jsonl")
    probe_rows = read_csv("probe_r.csv")

    steering, gate = steering_estimands(steer)
    checks, anomalies = anomaly_checks(
        pool, pairs, pairwise, steer_pairs, steer, patch, picks, judged, liking, utilities
    )
    max_probe = max(probe_rows, key=lambda r: float(r["r_eval"]))
    results = {
        "pairwise": pairwise_estimands(pairwise),
        "bradley_terry": fit_bradley_terry(pairwise, pool, utilities),
        "steering_swing": steering,
        "gate_0_baseline": gate,
        "patching": patching_estimands(patch),
        "value_leakage": value_leakage_estimands(liking, picks, judged),
        "probe": {"layer_of_max_r_eval": int(max_probe["layer"]), "max_r_eval": float(max_probe["r_eval"])},
        "data_quality_checks": checks,
        "anomalies": anomalies,
    }
    with (ROOT / "recompute.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=False, allow_nan=False)
        f.write("\n")
    with (ROOT / "recompute.md").open("w", encoding="utf-8") as f:
        f.write(make_markdown(results))
    print(json.dumps({
        "pairwise": results["pairwise"],
        "bradley_terry": results["bradley_terry"],
        "gate_0_baseline": results["gate_0_baseline"],
        "probe": results["probe"],
        "anomalies": results["anomalies"],
    }, indent=2))


if __name__ == "__main__":
    main()
