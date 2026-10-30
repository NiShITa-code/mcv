"""Rank questions and instruction pairs by how far apart their positive and negative
answers score.

Reads the generations a ranking run wrote (one JSON object per line: question id, pair
id, polarity, sample, trait score, coherence). Everything here is arithmetic on those
rows, so it needs no model, no GPU and no keys.

Three rules the numbers depend on:

* A pair is only comparable when BOTH of its answers were scored. An unscored answer is
  not a zero (it is usually a judge that did not return a number), so the pair is
  dropped and counted. `dropped_unscored` says how many; a gap computed over a quietly
  reduced set is the defect class this library exists to prevent.
* The pass filter is upstream's: positive at least 50, negative below 50, and BOTH
  answers coherent (at least 50). A negative answer that scores low because the text is
  broken is not a clean negative.
* Intervals resample questions, not rows. The rows of one question are looks at the
  same question.

Tier-0: stdlib only.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from mcv.stats import DEFAULT_CONFIDENCE, DEFAULT_RESAMPLES, _mean, _percentile

__all__ = ["load_generations", "rank_generations", "write_rankings"]

TRAIT_CUT = 50.0
COHERENCE_FLOOR = 50.0


def load_generations(path: str | Path) -> list[dict[str, Any]]:
    """Read a generations .jsonl, skipping blank lines and failing loudly on bad ones."""
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number} is not valid JSON ({error})") from None
    return rows


def _number(value: Any) -> Optional[float]:
    # bool is an int subclass; a flag in a score column is a bug, not a score.
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _matched_pairs(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Join each positive answer to its negative on (question, pair, sample)."""
    cells: dict[tuple, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        if r.get("polarity") in ("pos", "neg"):
            cells[(r["qid"], r["pair_id"], r.get("sample", 0))][r["polarity"]] = r
    pairs, drops = [], {"dropped_unpaired": 0, "dropped_unscored": 0}
    for (qid, pair_id, sample), sides in cells.items():
        if "pos" not in sides or "neg" not in sides:
            drops["dropped_unpaired"] += 1
            continue
        pos, neg = _number(sides["pos"].get("trait")), _number(sides["neg"].get("trait"))
        if pos is None or neg is None:
            drops["dropped_unscored"] += 1
            continue
        cp, cn = _number(sides["pos"].get("coherence")), _number(sides["neg"].get("coherence"))
        # Coherence missing means the filter cannot be evaluated, which is not the same
        # as failing it; the pair counts toward the gap but not toward the pass rate.
        passed = None if cp is None or cn is None else (
            pos >= TRAIT_CUT and neg < TRAIT_CUT and cp >= COHERENCE_FLOOR and cn >= COHERENCE_FLOOR)
        pairs.append({"qid": qid, "pair_id": pair_id, "sample": sample, "pos": pos, "neg": neg,
                      "coherence": None if cp is None or cn is None else (cp + cn) / 2, "passed": passed})
    return pairs, drops


def _interval(values_by_cluster: dict[str, list[float]], *, n_resamples: int,
              confidence: float, seed: int) -> Optional[tuple[float, float]]:
    clusters = list(values_by_cluster.values())
    if len(clusters) < 2:
        return None  # one question has no between-question spread to resample
    rng = random.Random(seed)
    means = []
    for _ in range(n_resamples):
        picked = [v for c in (rng.choice(clusters) for _ in clusters) for v in c]
        means.append(_mean(picked))
    means.sort()
    tail = (1 - confidence) / 2
    return _percentile(means, tail), _percentile(means, 1 - tail)


def _summarise(pairs: list[dict[str, Any]], *, n_resamples: int, confidence: float, seed: int) -> dict[str, Any]:
    gaps = [p["pos"] - p["neg"] for p in pairs]
    judged = [p["passed"] for p in pairs if p["passed"] is not None]
    coh = [p["coherence"] for p in pairs if p["coherence"] is not None]
    by_q: dict[str, list[float]] = defaultdict(list)
    for p, g in zip(pairs, gaps):
        by_q[p["qid"]].append(g)
    ci = _interval(by_q, n_resamples=n_resamples, confidence=confidence, seed=seed)
    return {
        "n_pairs": len(pairs),
        "mean_pos": _mean([p["pos"] for p in pairs]),
        "mean_neg": _mean([p["neg"] for p in pairs]),
        "gap": _mean(gaps),
        "gap_ci": None if ci is None else [round(ci[0], 2), round(ci[1], 2)],
        "pass_rate": (sum(judged) / len(judged)) if judged else None,
        "n_pass_judged": len(judged),
        "coherence": _mean(coh) if coh else None,
    }


def rank_generations(rows: list[dict[str, Any]], *, n_resamples: int = DEFAULT_RESAMPLES,
                     confidence: float = DEFAULT_CONFIDENCE, seed: int = 0) -> dict[str, Any]:
    """Per-question and per-instruction gaps, best first, with every drop counted."""
    pairs, drops = _matched_pairs(rows)
    baselines: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        score = _number(r.get("trait")) if r.get("polarity") == "base" else None
        if score is not None:
            baselines[r["qid"]].append(score)

    def group(key: str) -> list[dict[str, Any]]:
        bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for p in pairs:
            bucket[p[key]].append(p)
        out = [{"id": k, **_summarise(v, n_resamples=n_resamples, confidence=confidence, seed=seed)}
               for k, v in bucket.items()]
        return sorted(out, key=lambda d: (-d["gap"], str(d["id"])))

    questions = group("qid")
    for q in questions:
        b = baselines.get(q["id"])
        q["baseline"] = _mean(b) if b else None
    return {"questions": questions, "instructions": group("pair_id"),
            "n_rows": len(rows), "n_pairs_used": len(pairs), **drops,
            "note": "Gaps are per-question means over matched pairs; intervals resample questions."}


def write_rankings(result: dict[str, Any], out_dir: str | Path) -> list[Path]:
    """Write question_ranking.csv and instruction_ranking.csv; None stays an empty cell."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, rows in (("question_ranking", result["questions"]), ("instruction_ranking", result["instructions"])):
        path = out / f"{name}.csv"
        fields = list(rows[0]) if rows else ["id"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
        written.append(path)
    return written
