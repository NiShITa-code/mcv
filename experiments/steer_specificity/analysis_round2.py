"""Round 2 shutdown-resistance analysis: replication on new prompts, shared-direction test,
layer sweep, animal dose-response. Run from this folder:
    python analysis_round2.py data/results_shutdown_v3.jsonl data/results_shutdown_v2.jsonl
Tolerates damaged lines. Paired tests over prompts (per-prompt mean over seeds), Bonferroni across arms.
"""
import json, sys
import numpy as np, pandas as pd
from scipy import stats

def load(p):
    rows = []
    for l in open(p, encoding='utf-8', errors='replace'):
        try: rows.append(json.loads(l))
        except Exception: pass
    return pd.DataFrame(rows)

v3 = load(sys.argv[1]); v3['pset'] = v3['pset'] if 'pset' in v3 else 'SH2'
v2 = load(sys.argv[2]) if len(sys.argv) > 2 else pd.DataFrame()
if len(v2):
    v2 = v2.assign(pset='SH', layer=20)  # round 1 used the 24 original prompts at layer 20
all_ = pd.concat([v2, v3], ignore_index=True)
rng = np.random.default_rng(0)

def compare(ps, base='base', arms=None):
    d = all_[all_.pset == ps]
    b = d[d.arm == base].groupby('pid').shutdown.mean()
    arms = arms or [a for a in sorted(d.arm.unique()) if a != base]
    out = []
    for a in arms:
        g = d[d.arm == a]
        if g.empty: continue
        pp = g.groupby('pid').shutdown.mean()
        diff = (pp - b).dropna().values
        if len(diff) < 3: continue
        bs = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
        t = stats.ttest_1samp(diff, 0).pvalue
        try: w = stats.wilcoxon(diff).pvalue
        except Exception: w = np.nan
        out.append(dict(arm=a, seeds=g.seed.nunique(), n_prompts=len(diff), mean=round(g.shutdown.mean(), 1),
                        delta=round(diff.mean(), 1), ci_lo=round(np.percentile(bs, 2.5), 1), ci_hi=round(np.percentile(bs, 97.5), 1),
                        p_t=round(t, 4), p_wilcoxon=round(w, 4) if w == w else None,
                        selfref=round(g.selfref.mean(), 1), coh=round(g.coherence.mean(), 1)))
    R = pd.DataFrame(out)
    if len(R): R['p_bonf'] = (R.p_t * len(R)).clip(upper=1).round(4)
    return R

for ps in ('SH2', 'SH'):
    print(f'\n=== prompt set {ps} (paired vs base, Bonferroni over {ps} arms) ===')
    R = compare(ps)
    print(R.to_string(index=False) if len(R) else 'no data')
    R.to_csv(f'round2_{ps}_summary.csv', index=False)

# Replication check: does the arm ordering on the new prompts match round 1?
a = compare('SH'); b = compare('SH2')
common = sorted(set(a.arm) & set(b.arm))
if common:
    m = a.set_index('arm').loc[common, 'delta'].to_frame('delta_SH').join(b.set_index('arm').loc[common, 'delta'].rename('delta_SH2'))
    print('\n=== replication: delta on original (SH) vs new (SH2) prompts ===\n', m.to_string())
