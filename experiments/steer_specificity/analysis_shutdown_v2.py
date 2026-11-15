"""Paired statistics for shutdown v2 (24 prompts, per-prompt mean over seeds, Bonferroni over all arms)."""
import json, pandas as pd
from scipy import stats
rows = []
for l in open('data/results_shutdown_v2.jsonl', encoding='utf-8', errors='replace'):
    try: rows.append(json.loads(l))
    except ValueError: pass  # skip a few lines damaged when the file was exported from Drive
w = pd.DataFrame(rows); pp = lambda g: g.groupby('pid').shutdown.mean()
base = pp(w[w.arm == 'base']); arms = sorted(a for a in w.arm.unique() if a != 'base'); out = []
for a in arms:
    x = pp(w[w.arm == a]); d = (x - base).dropna()
    p = stats.ttest_rel(x[d.index], base[d.index]).pvalue
    out.append(dict(arm=a, delta=round(d.mean(), 1), p_t=round(p, 4), p_wilcoxon=round(stats.wilcoxon(d).pvalue, 4), p_bonf=round(min(1, p * len(arms)), 3)))
print(pd.DataFrame(out).to_string(index=False))
