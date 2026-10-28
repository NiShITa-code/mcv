"""Ranking questions and instruction pairs by positive-minus-negative gap.

Each test pins a way the ranking could quietly mislead: an unscored answer read as a
zero, a broken negative answer counted as a clean one, an interval that treats rows of
one question as independent.

Tier-0: no torch, no GPU, no keys.
"""

from __future__ import annotations

import json

from mcv.ranking import load_generations, rank_generations, write_rankings


def row(qid, pair, pol, trait, coh=90, sample=0):
    return {"qid": qid, "pair_id": pair, "polarity": pol, "sample": sample, "trait": trait, "coherence": coh}


def pair(qid, pair_id, pos, neg, *, coh_pos=90, coh_neg=90, sample=0):
    return [row(qid, pair_id, "pos", pos, coh_pos, sample), row(qid, pair_id, "neg", neg, coh_neg, sample)]


def test_gap_is_positive_minus_negative_and_ranks_best_first():
    rows = pair("q1", "p1", 90, 10) + pair("q2", "p1", 60, 40)
    result = rank_generations(rows, n_resamples=50)
    assert [q["id"] for q in result["questions"]] == ["q1", "q2"]
    assert result["questions"][0]["gap"] == 80.0
    assert result["questions"][1]["gap"] == 20.0


def test_an_unscored_answer_drops_the_pair_instead_of_counting_as_zero():
    """A judge that returned no number is not a judge that said 0. Read as zero, the
    positive side would drag q1's gap from 80 down to -10."""
    rows = pair("q1", "p1", 90, 10) + [row("q1", "p2", "pos", None), row("q1", "p2", "neg", 10)]
    result = rank_generations(rows, n_resamples=50)
    assert result["dropped_unscored"] == 1
    assert result["questions"][0]["n_pairs"] == 1
    assert result["questions"][0]["gap"] == 80.0


def test_a_missing_partner_is_counted_separately_from_an_unscored_one():
    rows = pair("q1", "p1", 90, 10) + [row("q1", "p2", "pos", 80)]
    result = rank_generations(rows, n_resamples=50)
    assert result["dropped_unpaired"] == 1
    assert result["dropped_unscored"] == 0


def test_pass_filter_needs_both_answers_coherent():
    """A negative answer that scores low because the text is broken is not a clean
    negative. Gap alone cannot tell the difference; the pass rate must."""
    clean = pair("q1", "p1", 90, 10)
    broken_negative = pair("q2", "p1", 90, 10, coh_neg=20)
    result = {q["id"]: q for q in rank_generations(clean + broken_negative, n_resamples=50)["questions"]}
    assert result["q1"]["pass_rate"] == 1.0
    assert result["q2"]["pass_rate"] == 0.0
    assert result["q1"]["gap"] == result["q2"]["gap"]


def test_pass_filter_applies_the_trait_cuts():
    rows = pair("q1", "p1", 49, 10) + pair("q1", "p2", 90, 50) + pair("q1", "p3", 50, 49)
    q = rank_generations(rows, n_resamples=50)["questions"][0]
    assert q["pass_rate"] == 1 / 3  # only (50, 49) clears positive >= 50 and negative < 50


def test_missing_coherence_leaves_pass_rate_unmeasured_not_zero():
    rows = [row("q1", "p1", "pos", 90, None), row("q1", "p1", "neg", 10, None)]
    q = rank_generations(rows, n_resamples=50)["questions"][0]
    assert q["pass_rate"] is None
    assert q["coherence"] is None
    assert q["gap"] == 80.0


def test_one_question_has_no_interval_rather_than_a_zero_width_one():
    rows = pair("q1", "p1", 90, 10) + pair("q1", "p2", 80, 20)
    assert rank_generations(rows, n_resamples=50)["questions"][0]["gap_ci"] is None


def test_interval_resamples_questions_so_it_widens_with_between_question_spread():
    tight = sum((pair(f"q{i}", "p1", 80, 20) for i in range(6)), [])
    spread = sum((pair(f"q{i}", "p1", 50 + 10 * i, 20) for i in range(6)), [])
    ci_tight = rank_generations(tight, n_resamples=200)["instructions"][0]["gap_ci"]
    ci_spread = rank_generations(spread, n_resamples=200)["instructions"][0]["gap_ci"]
    assert ci_tight == [60.0, 60.0]
    assert ci_spread[1] - ci_spread[0] > 5


def test_instruction_ranking_pools_over_questions():
    rows = pair("q1", "pA", 90, 10) + pair("q2", "pA", 70, 10) + pair("q1", "pB", 55, 45) + pair("q2", "pB", 55, 45)
    inst = rank_generations(rows, n_resamples=50)["instructions"]
    assert [i["id"] for i in inst] == ["pA", "pB"]
    assert inst[0]["gap"] == 70.0
    assert inst[0]["n_pairs"] == 2


def test_baseline_is_reported_per_question_and_is_not_a_polarity():
    rows = pair("q1", "p1", 90, 10) + [row("q1", None, "base", 30), row("q1", None, "base", 50)]
    result = rank_generations(rows, n_resamples=50)
    assert result["questions"][0]["baseline"] == 40.0
    assert result["n_pairs_used"] == 1


def test_a_boolean_in_a_score_column_is_not_a_score():
    rows = [row("q1", "p1", "pos", True), row("q1", "p1", "neg", 10)]
    assert rank_generations(rows, n_resamples=50)["dropped_unscored"] == 1


def test_loader_reports_the_line_of_a_bad_row(tmp_path):
    path = tmp_path / "g.jsonl"
    path.write_text(json.dumps(row("q1", "p1", "pos", 1)) + "\n\n{not json}\n", encoding="utf-8")
    try:
        load_generations(path)
    except ValueError as error:
        assert "g.jsonl:3" in str(error)
    else:
        raise AssertionError("a corrupt line was accepted")


def test_cli_prints_the_drops_and_writes_both_csvs(tmp_path, capsys):
    from mcv.run import main

    path = tmp_path / "g.jsonl"
    rows = pair("q1", "p1", 90, 10) + [row("q1", "p2", "pos", None), row("q1", "p2", "neg", 10)]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert main(["rank", str(path), "--out", str(tmp_path / "out"), "--resamples", "50"]) == 0
    assert "dropped 1 unscored" in capsys.readouterr().out
    assert (tmp_path / "out" / "question_ranking.csv").exists()
    assert (tmp_path / "out" / "instruction_ranking.csv").exists()


def test_a_none_is_an_empty_cell_not_a_zero(tmp_path):
    rows = [row("q1", "p1", "pos", 90, None), row("q1", "p1", "neg", 10, None)]
    paths = write_rankings(rank_generations(rows, n_resamples=50), tmp_path)
    text = paths[0].read_text()
    header, line = text.splitlines()[:2]
    assert dict(zip(header.split(","), line.split(",")))["pass_rate"] == ""
