# mcv

mcv runs reproducible persona-vector steering experiments.

mcv wraps [`safety-research/persona_vectors`](https://github.com/safety-research/persona_vectors).
An experiment is a config file, not a code edit. A run that dies with its pod still
leaves a record of what it was.

```bash
mcv validate mcv/configs/repro-evil-extract.yaml   # laptop, no GPU, no keys
mcv extract  mcv/configs/repro-evil-extract.yaml   # build a trait vector
mcv sweep    mcv/configs/repro-evil-sweep.yaml     # dose-response curve
```

Start at [Getting started](getting-started.md). For one specific problem, read
[How-to](how-to.md). For the vocabulary, read [Concepts](concepts.md).

## What mcv guarantees

- **A record before the run.** mcv writes the manifest before anything runs. The
  manifest holds the config, the seed, the git commit, `git_dirty` and the destination.
  A pod killed one second in still leaves a record. mcv logs failures and kills, not
  only successes.
- **Artifacts survive the pod.** mcv pushes every output off the pod as it produces it.
  mcv checksums each file at both ends. An unsaved artifact is a lost run.
- **Upstream stays unmodified.** mcv changes no upstream source file. Your results stay
  comparable with the paper's.
- **No hidden choices.** Three settings change what a number means: the engine, the
  steering position and vector normalisation. Each one is a config field. Each one
  defaults to upstream's own behaviour.

## Contributors

Aaina Garg, Kaustubh Gupta, Nishita Jain, Jacqueline Payne.

Built for the Moral Consideration Vectors arm of the Sentient Futures Project
Incubator, with Compassion in Machine Learning (CaML). Per-file credits are in
[NOTICE](https://github.com/NiShITa-code/mcv/blob/main/NOTICE).
