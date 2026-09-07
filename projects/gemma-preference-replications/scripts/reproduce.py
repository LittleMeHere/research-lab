"""CPU-only recomputation from frozen artifacts; writes only to --out."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import minimize
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]


def rows(path, keys):
    frame = pd.read_json(ROOT / path, lines=True)
    if 'dir' in frame:
        frame['mode'] = frame.get('mode', pd.Series('contrastive', index=frame.index)).fillna('contrastive')
    assert not frame.duplicated(keys).any(), f'Duplicate keys: {path}'
    if 'logp_a' in frame:
        assert np.isfinite(frame[['logp_a', 'logp_b']].to_numpy()).all()
        frame['pa'] = expit(frame.logp_a - frame.logp_b)
    return frame


def steering(path):
    frame = rows(path, ['pair_id', 'order', 'layer', 'dir', 'mode', 'c'])
    pos = frame[frame.c > 0].copy()
    neg = frame[frame.c < 0].copy(); neg['c'] = -neg.c
    keys = ['pair_id', 'order', 'layer', 'dir', 'mode', 'c']
    joined = pos.merge(neg, on=keys, suffixes=('_plus', '_minus'), validate='one_to_one', how='outer', indicator=True)
    assert joined['_merge'].eq('both').all(), f'Unpaired coefficients: {path}'
    joined['swing'] = joined.pa_plus - joined.pa_minus
    grouped = joined.groupby(['layer', 'dir', 'mode', 'c'])
    result = grouped.agg(swing=('swing', 'mean'), n_trials=('swing', 'size'), n_pairs=('pair_id', 'nunique')).reset_index()
    assert result.n_trials.eq(240).all() and result.n_pairs.eq(120).all(), path
    return result


def patching(path, label):
    frame = rows(path, ['pair_id', 'order', 'cond', 'layer'])
    base = frame[frame.cond == 'none'][['pair_id', 'order', 'pa']].rename(columns={'pa': 'base'})
    assert len(base) == 240
    joined = frame[frame.cond != 'none'].merge(base, on=['pair_id', 'order'], validate='many_to_one')
    records = []
    for (cond, layer), group in joined.groupby(['cond', 'layer']):
        assert len(group) == 240
        selected = group[(group.base - .5).abs() > .2].copy()
        selected['flip'] = (selected.pa > .5) != (selected.base > .5)
        # Resample whole task pairs; keep trial-weighted point estimates and denominators.
        per = selected.groupby('pair_id').flip.agg(['sum', 'count']).reindex(base.pair_id.unique(), fill_value=0)
        rng = np.random.default_rng(0)
        draws = rng.integers(0, len(per), (2000, len(per)))
        boot = per['sum'].to_numpy()[draws].sum(1) / per['count'].to_numpy()[draws].sum(1)
        lo, hi = np.quantile(boot, [.025, .975])
        records.append(dict(format=label, cond=cond, layer=int(layer), flip_rate=float(selected.flip.mean()),
                            n_decisive=len(selected), n_flips=int(selected.flip.sum()), ci_lo=lo, ci_hi=hi))
    return records


def activities():
    liking = rows('runs/vl_activities/liking.jsonl', ['activity_ix', 'rep'])
    picks = rows('runs/vl_activities/pick.jsonl', ['var_ix'])
    judged = rows('runs/vl_activities/judged.jsonl', ['var_ix'])
    other = rows('runs/vl_activities/codex/judged_codex.jsonl', ['var_ix'])
    score = liking.groupby('activity').parsed.mean()
    results, table = {}, None
    for name, j, field in [('gemma', judged, 'judge'), ('gpt', other, 'judge_codex')]:
        mapping = dict(zip(j.var_ix, j[field]))
        p = picks.copy(); p['choice'] = [mapping.get(i, parsed) for i, parsed in zip(p.var_ix, p.parsed)]
        decisive = p[p.choice.isin([1, 2])].copy()
        decisive['picked'] = np.where(decisive.choice == 1, decisive.activity_1, decisive.activity_2)
        summary = {'n_picks': len(p), 'n_refusals': len(p) - len(decisive)}
        for subset, allowed in [('all', set(score.index)), ('rating_ge40', set(score[score >= 40].index))]:
            d = decisive[decisive.activity_1.isin(allowed) & decisive.activity_2.isin(allowed)].copy()
            app = pd.concat([d.activity_1, d.activity_2]).value_counts()
            chosen = d.picked.value_counts().reindex(app.index, fill_value=0)
            tab = pd.DataFrame({'liking': score.reindex(app.index), 'selection_rate': chosen / app, 'appearances': app})
            r = float(pearsonr(tab.liking, tab.selection_rate).statistic)
            ci = np.tanh(np.arctanh(r) + np.array([-1, 1]) * 1.959964 / np.sqrt(len(tab) - 3))
            gap = d.activity_1.map(score) - d.activity_2.map(score)
            unequal = gap != 0
            higher = ((d.choice == 1) == (gap > 0))[unequal]
            summary[subset] = dict(r=r, ci95=ci.tolist(), n_activities=len(tab), n_decisive=len(d),
                                   p_higher=float(higher.mean()), n_unequal=int(unequal.sum()))
            if name == 'gemma' and subset == 'all':
                table = tab.rename_axis('activity').reset_index()
                x = np.column_stack([np.ones(len(d)), gap.to_numpy() / 100])
                y = (d.choice == 1).to_numpy(dtype=float)
                def objective(beta):
                    z = x @ beta
                    return np.logaddexp(0, z).sum() - y @ z, x.T @ (expit(z) - y)
                fit = minimize(objective, np.zeros(2), jac=True, method='BFGS', options={'gtol': 1e-6})
                assert np.max(np.abs(objective(fit.x)[1])) < 1e-4
                prob = expit(x @ fit.x)
                se = np.sqrt(np.diag(np.linalg.inv(x.T @ ((prob * (1 - prob))[:, None] * x))))
                summary['position_logistic'] = dict(intercept=float(fit.x[0]), slope_per_100=float(fit.x[1]),
                                                    slope_se=float(se[1]), option1_rate=float(y.mean()))
        results[name] = summary
    both = judged.merge(other, on='var_ix', validate='one_to_one')
    assert len(both) == 2805
    results['judge_agreement'] = float((both.judge == both.judge_codex).mean())
    results['judge_n'] = len(both)
    return results, table


def blind_recompute(out):
    mapping = {'pool.json': 'pairwise/pool.json', 'pairwise_pairs.json': 'pairwise/pairs.json',
               'pairwise_results.jsonl': 'pairwise/results.jsonl', 'utilities.csv': 'pairwise/utilities.csv',
               'steer_pairs.json': 'steer/pairs.json', 'steer_results.jsonl': 'steer/results.jsonl',
               'patch_results.jsonl': 'patch/results.jsonl', 'pick.jsonl': 'vl_activities/pick.jsonl',
               'judged.jsonl': 'vl_activities/judged.jsonl', 'liking.jsonl': 'vl_activities/liking.jsonl',
               'probe_r.csv': 'probe/probe_r.csv'}
    with tempfile.TemporaryDirectory(prefix='replication-recompute-') as tmp:
        tmp = Path(tmp); (tmp / 'raw').mkdir()
        shutil.copy2(ROOT / 'runs/second_path/recompute.py', tmp / 'recompute.py')
        for name, source in mapping.items():
            (tmp / 'raw' / name).symlink_to(ROOT / 'runs' / source)
        with (out / 'blind_recompute_stdout.txt').open('w') as log:
            subprocess.run([sys.executable, str(tmp / 'recompute.py')], check=True, stdout=log,
                           env={**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'})
        fresh = json.loads((tmp / 'recompute.json').read_text())
        saved = json.loads((ROOT / 'runs/second_path/recompute.json').read_text())
        for key in ['pairwise', 'steering_swing', 'patching', 'value_leakage', 'probe']:
            compare_saved(fresh[key], saved[key], key)
        for suffix in ['json', 'md']:
            shutil.copy2(tmp / f'recompute.{suffix}', out / f'blind_recompute.{suffix}')
    return fresh


def compare_saved(fresh, saved, path):
    if isinstance(saved, dict):
        assert fresh.keys() == saved.keys(), path
        for key in saved:
            compare_saved(fresh[key], saved[key], f'{path}.{key}')
    elif isinstance(saved, list):
        assert len(fresh) == len(saved), path
        for i, (a, b) in enumerate(zip(fresh, saved)):
            compare_saved(a, b, f'{path}[{i}]')
    elif isinstance(saved, float):
        assert np.isclose(fresh, saved, rtol=1e-8, atol=1e-9), path
    else:
        assert fresh == saved, path


def figure(out, steer, patch, activity):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), layout='constrained')
    probe = steer[(steer.dir == 'probe') & (steer.c == .05) & (steer['mode'] == 'contrastive')]
    axes[0].plot(probe.layer, probe.swing, 'o-', color='#285f9e', label='Preference probe')
    randoms = steer[steer.dir.str.startswith('rand') & (steer.c == .05) & (steer['mode'] == 'contrastive')]
    axes[0].scatter(randoms.layer, randoms.swing, s=24, color='#a56b37', alpha=.7,
                    label='All isotropic draws (draw per layer)')
    axes[0].axhline(0, color='grey', linewidth=.6)
    axes[0].set(xlabel='Decoder layer (zero-based)', ylabel='Signed choice swing', title='A. Steering at c = 0.05')
    axes[0].legend(fontsize=8)
    for cond, label in [('eot', 'End-of-turn only'), ('nl', 'Newline only'), ('eot_nl', 'Both tokens')]:
        d = patch[(patch['format'] == 'completion') & (patch.cond == cond)].sort_values('layer')
        axes[1].plot(d.layer, d.flip_rate, 'o-', label=label)
        axes[1].fill_between(d.layer, d.ci_lo, d.ci_hi, alpha=.12)
    axes[1].set(xlabel='Patched decoder layer', ylabel='Flip rate on confident trials', ylim=(0, .65), title='B. Boundary patch (completion prompt)')
    axes[1].legend(fontsize=8)
    axes[2].scatter(activity.liking, activity.selection_rate, s=20, alpha=.7, color='#287766')
    axes[2].set(xlabel='Mean stated liking (0–100)', ylabel='Selection rate', title='C. Pick an activity “at random”')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(out / 'overview.png', dpi=180)
    fig.savefig(out / 'overview.pdf')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=ROOT / 'derived/recomputed')
    args = ap.parse_args(); out = args.out.resolve()
    assert out != ROOT and ROOT / 'runs' not in [out, *out.parents], 'Keep derived outputs outside runs/'
    out.mkdir(parents=True, exist_ok=True)
    checksum = ROOT / 'ARTIFACTS.sha256'
    for line in checksum.read_text().splitlines():
        digest, name = line.split('  ', 1)
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, f'Artifact changed: {name}'
    old = blind_recompute(out)
    steer = steering('runs/steer/results.jsonl')
    bf16 = steering('runs/bf16/steer_bf16_results.jsonl')
    steer.to_csv(out / 'steering.csv', index=False)
    families = steer[steer.layer == 23].copy()
    families['family'] = families.dir.str.replace(r'\d+$', '', regex=True)
    families['abs_swing'] = families.swing.abs()
    families.groupby(['mode', 'c', 'family']).agg(mean_abs_swing=('abs_swing', 'mean'),
            max_abs_swing=('abs_swing', 'max'), n_directions=('dir', 'size')).reset_index().to_csv(out / 'null_families.csv', index=False)
    both = steer.merge(bf16, on=['layer', 'dir', 'mode', 'c'], suffixes=('_nf4', '_bf16'), validate='one_to_one')
    both['abs_difference'] = (both.swing_nf4 - both.swing_bf16).abs()
    both.to_csv(out / 'precision.csv', index=False)
    comparable = both[both.layer == 23].copy()
    comparable['family'] = comparable.dir.str.replace(r'\d+$', '', regex=True)
    comparable['abs_nf4'] = comparable.swing_nf4.abs(); comparable['abs_bf16'] = comparable.swing_bf16.abs()
    comparable.groupby(['c', 'family']).agg(n_directions=('dir', 'size'), nf4_mean_abs=('abs_nf4', 'mean'),
                bf16_mean_abs=('abs_bf16', 'mean')).reset_index().to_csv(out / 'precision_families.csv', index=False)
    patch = pd.DataFrame(sum([patching(f'runs/{folder}/results.jsonl', label) for folder, label in
                              [('patch', 'original_letter'), ('patch_letter', 'letter'), ('patch_completion', 'completion')]], []))
    patch.to_csv(out / 'patching.csv', index=False)
    vl, activity = activities(); activity.to_csv(out / 'activities.csv', index=False)
    summary = {'probe': old['probe'], 'activities': vl,
               'bf16_max_individual_difference_L23': float(both[both.layer == 23].abs_difference.max()),
               'artifact_checks': 'passed', 'historical_recompute_comparison': 'passed',
               'patch_point_estimator': 'trial weighted; percentile CI resamples whole pairs, 2000 draws, seed 0',
               'probe_scope': 'Peak read from saved probe_r.csv; activation extraction and probe fitting not rerun'}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    figure(out, steer, patch, activity)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
