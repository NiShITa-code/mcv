# mcv

Run persona-vector steering experiments from a config file, and get the same numbers
back when you run them again.

mcv wraps [`safety-research/persona_vectors`](https://github.com/safety-research/persona_vectors)
(Chen et al. 2025, [arXiv:2507.21509](https://arxiv.org/abs/2507.21509)). It changes no
upstream source file.

```bash
mcv validate mcv/configs/repro-evil-extract.yaml   # laptop, no GPU, no keys
mcv extract  mcv/configs/repro-evil-extract.yaml   # build a trait vector
mcv sweep    mcv/configs/repro-evil-sweep.yaml     # dose-response curve
```

Start at [Getting started](getting-started.md). For one specific problem, read
[How-to](how-to.md). For the vocabulary, read [Concepts](concepts.md).

## Why not call persona_vectors directly

Four things in upstream change what your numbers mean. Every one is cited, so you can
check it before you trust us. mcv turns each into a config field that defaults to
upstream's own behaviour, so you opt in rather than get surprised.

| in upstream `b8e0f04` | what it costs you | in mcv |
|---|---|---|
| `random.choices` picks the paraphrases and `SamplingParams` carries no seed (`eval/eval_persona.py:109`, `:59`) | the extraction set differs on every run, so a vector cannot be rebuilt from its config | `seed` seeds the generation path, not just training |
| `if coef != 0` sends steered runs to HF and unsteered runs to vLLM (`eval/eval_persona.py:117`, `:178`, `:256`) | your baseline and your steered arm ran on different engines, so part of the difference is the engine | `engine` is one field, used on both sides of the comparison |
| `steer = self.coeff * self.vector`, with no normalisation (`activation_steer.py:76`) | coefficient 2 is a different intervention on every vector, so coefficients do not transfer between traits or layers | vectors are normalised and the raw norm is recorded beside them |
| `dataset.train_test_split(test_size=0.1)` takes no seed (`training.py:226`) | held-out loss is measured against a different split each run | the split is seeded and the holdout is an explicit config field |

[Concepts](concepts.md) explains each of these as a term you will meet in a config file.

## If you run on rented GPUs

There is a fifth. Upstream defines `push_model` (`training.py:248`) and never calls it,
and its output directories are gitignored. An eval transcript written to a pod is gone
when the pod is terminated.

mcv writes the manifest — config, seed, git commit, dirty flag, destination — *before*
the run starts, and pushes each artifact off-box as it is produced, checksummed at both
ends. A pod killed one second in still leaves a record of what it was.

## Contributors

Aaina Garg, Kaustubh Gupta, Nishita Jain, Jacqueline Payne.

Built for the Moral Consideration Vectors arm of the Sentient Futures Project
Incubator, with Compassion in Machine Learning (CaML). Per-file credits are in
[NOTICE](https://github.com/NiShITa-code/mcv/blob/main/NOTICE).
