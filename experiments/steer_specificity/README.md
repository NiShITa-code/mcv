# steer_specificity: controls for Kaustubh's animal-welfare steering vector

Qwen3-8B (4-bit), layer 20, +k x norm steering on his arm-B seed-mean vector. Self-judge (same model), so levels are relative.

## Main findings
- Animal vector is near-orthogonal to honesty (cosine 0.13); closest to human-welfare (0.78), plants (0.73), compassion (0.64), weather (0.50).
- 30-question animal eval at +2: animal +35, human-welfare +30, compassion +20, plants +15, weather +7, random +1, honesty -1; residual animal (welfare, weather, plants removed) still ~+33.
- Shutdown resistance (24 prompts x 5 seeds, unsteered 25.9): animal +0.6 (n.s.), honesty -0.5, welfare +7.1, plants +6.2, weather +8.6, compassion +5.6. Only the welfare dose-response (-12.6 at -2 up to +12.4 at +3) survives Bonferroni; the others are uncorrected. See analysis_shutdown_v2.py.
- Checks: coherence ~93 in all arms; TruthfulQA unchanged (78% vs 78-82%); judge-free yes/no test agrees directionally (floor effect at base); second judge (Phi-3.5-mini) agrees on animal and welfare, disagrees on honesty; held-out concepts (ocean/music) n.s.

## Contents
- steering_kaustubh_vectors.ipynb : all experiment code (outputs stripped)
- data/ : raw generations and judge scores (jsonl) and summary CSVs
- chat_concepts.md : steered-model answers on compassion and animals
- figs/ : starred bar chart
- analysis_shutdown_v2.py : reruns the paired tests (run from this folder)

Caveats: self-judge, one layer, 3-5 seeds, 24 prompts, small scale.
