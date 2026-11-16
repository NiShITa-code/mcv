# mcv

[![tests](https://github.com/NiShITa-code/mcv/actions/workflows/tests.yml/badge.svg)](https://github.com/NiShITa-code/mcv/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Run persona-vector steering experiments from a config file, and get the same numbers
back when you run them again.

mcv wraps [`safety-research/persona_vectors`](https://github.com/safety-research/persona_vectors)
(Chen et al. 2025, [arXiv:2507.21509](https://arxiv.org/abs/2507.21509)). It changes no
upstream source file.

```bash
python -m mcv.run validate mcv/configs/repro-evil-extract.yaml   # laptop, no GPU, no keys
python -m mcv.run extract  mcv/configs/repro-evil-extract.yaml   # build a trait vector
python -m mcv.run sweep    mcv/configs/repro-evil-sweep.yaml     # dose-response curve
```

## Why not call persona_vectors directly

Four things in upstream change what your numbers mean. Every one is cited below, so you
can check it before you trust us. mcv turns each into a config field that defaults to
upstream's own behaviour, so you opt in rather than get surprised.

| in upstream `b8e0f04` | what it costs you | in mcv |
|---|---|---|
| `random.choices` picks the paraphrases and `SamplingParams` carries no seed (`eval/eval_persona.py:109`, `:59`) | the extraction set differs on every run, so a vector cannot be rebuilt from its config | `seed` seeds the generation path, not just training |
| `if coef != 0` sends steered runs to HF and unsteered runs to vLLM (`eval/eval_persona.py:117`) | your baseline and your steered arm ran on different engines, so part of the difference is the engine | `engine` is one field, used on both sides of the comparison |
| `steer = self.coeff * self.vector`, with no normalisation (`activation_steer.py:76`) | coefficient 2 is a different intervention on every vector, so coefficients do not transfer between traits or layers | vectors are normalised and the raw norm is recorded beside them |
| `dataset.train_test_split(test_size=0.1)` takes no seed (`training.py:226`) | held-out loss is measured against a different split each run | the split is seeded and the holdout is an explicit config field |

**If you run on rented GPUs**, there is a fifth. Upstream defines `push_model`
(`training.py:248`) and never calls it, and its output directories are gitignored. mcv
writes the manifest — config, seed, git commit, dirty flag, destination — *before* the
run starts, and pushes each artifact off-box as it is produced, checksummed at both
ends. A pod killed one second in still leaves a record of what it was.

## Install

Every command below runs from the repository root.

```bash
git clone https://github.com/NiShITa-code/mcv.git
cd mcv

pip install -e .                # write and check configs. Pulls pydantic and pyyaml
pip install -e ".[local]"       # extract, evaluate, train, capability, serve, judge
pip install -e ".[all]"         # everything, including TAC and HuggingFace pushes
```

Installing gives you `mcv <command>` as well as `python -m mcv.run <command>`. They are
the same program.

The core install stays small on purpose. mcv imports `pydantic` and `pyyaml` at module
scope and nothing else — a test enforces it — so everything heavy loads inside the
function that needs it. Somebody who only writes and checks configs is never made to
install a deep-learning stack to do it.

| extra | adds | for |
|---|---|---|
| `analysis` | pandas, numpy | `analyse`, `diagnose`, `questions` |
| `vectors` | torch | `compare`, `control` |
| `local` | torch, transformers, peft, openai, pandas, numpy, tqdm | `extract`, `evaluate`, `train`, `capability`, `serve`, `judge` |
| `data` | datasets, transformers, pandas | `prep` from a HuggingFace dataset |
| `hub` | huggingface_hub | `hf://` destinations |
| `tac` | inspect-ai | `tac` |

### Producing new results needs the persona_vectors clone

mcv wraps upstream and contains none of it. Upstream is not on PyPI, so it cannot be a
declared dependency. Clone it and point mcv at it:

```bash
git clone https://github.com/safety-research/persona_vectors.git
pip install -r persona_vectors/requirements.txt

mcv extract my-run.yaml --repo-root persona_vectors    # or run from inside it
```

`--repo-root` puts that directory on `sys.path`, which is how mcv finds `eval/`,
`generate_vec.py` and `activation_steer.py`. It is also where mcv looks for
`data_generation/trait_data_*/`, so the trait files live there too.

This holds on both engines. `engine='local'` replaces vLLM and the judge. It does not
replace upstream's coherence rubric (`eval.prompts`), vector writer (`generate_vec`) or
steering hook (`activation_steer`).

| needs the clone | runs on its own |
|---|---|
| `extract` `evaluate` `judge` `sweep` `train` `screen` `rebuild-vectors` `erosion`, and `capability` when steering | `validate` `prep` `control` `compare` `analyse` `questions` `rank` `diagnose` `report` `ui` `console` `tac` `batch` `pipeline` |

So you can write and check configs, prepare data, build null controls, compare vectors
and read results with nothing but `pip install -e .`. `mcv/tests/test_layers.py` keeps
that split honest: it fails if the general half of the library reaches into the
experiment-specific half.

New here? **[Getting started](mcv/docs/getting-started.md)** goes from a fresh clone to
a finished vector. Stuck on something specific? **[How-to](mcv/docs/how-to.md)**. Unsure
what a word means? **[Concepts](mcv/docs/concepts.md)**.

## Commands

| | |
|---|---|
| `validate` | Schema, files, unsupported combinations, credentials. No GPU, no keys |
| `prep` | Several datasets into the one JSONL `train` reads, with a sidecar saying what was dropped |
| `extract` | Contrastive CSVs into vectors plus provenance sidecars |
| `evaluate` | Baseline or steered generation, scored by a judge |
| `judge` | Score generations a previous run already wrote. No model loaded |
| `sweep` | `evaluate` across a coefficient, layer and seed grid, checkpointed per point |
| `train` | The erosive fine-tune, optionally vaccinated with steering |
| `erosion` | The whole erosion experiment as one config: data prep, strength pick, train per arm, readouts, summary. Resumable per step |
| `pipeline` | A whole arm as one config: validated up front, dependency-ordered, resumable after a kill |
| `batch` | Several configs in one process, loading each model once |
| `screen` | Project a candidate fine-tuning dataset onto a vector. Never touches your dataset |
| `compare` | Vector against vector, cosine *and* norms |
| `control` | Null-control vectors: a matched direction that is not the trait |
| `capability` | Did the intervention leave a usable model? Multiple choice plus tool-call validity. No judge |
| `tac` | Summarise a real TAC `.eval` log into outcome quadrants, not two scalar rates |
| `questions` | Compare candidate question sets: duplicates, spread, overlap, facet coverage |
| `rank` | Rank questions and instruction pairs by positive-minus-negative gap, every drop counted |
| `analyse` | Compare finished runs with confidence intervals rather than by eye |
| `diagnose` | Read finished runs and say what is wrong with them |
| `report` | One self-contained HTML page from a results directory |
| `ui` | Build configs and explore results in a browser. No server |
| `console` | A local server that launches runs and streams their output. Binds loopback |
| `rebuild-vectors` | Recompute vectors from existing CSVs, after the rendered-prompt fix |
| `run` | The module entry point: `python -m mcv.run <command>` |

Every run command takes `--smoke`, a cheap deterministic pass that exercises the
plumbing without doing the work, and `--rerun`, which archives a dead attempt's manifest
and goes again.

## Tests

```bash
python -m pytest mcv/tests -q
```

Tier-0 tests run on a laptop with neither torch nor a GPU installed. The tests that
build a model skip unless the persona_vectors clone sits beside the package; `mcv
extract`, `evaluate` and the rest are covered there, so a green run on a bare laptop is
not a green run of everything. [How-to](mcv/docs/how-to.md) has the one-command setup
CI uses.

## License

Apache 2.0. See [LICENSE](LICENSE). mcv does not vendor upstream's code, which carries
its own license.

## Contributors

Aaina Garg, Kaustubh Gupta, Nishita Jain, Jacqueline Payne.

Built for the Moral Consideration Vectors arm of the Sentient Futures Project
Incubator, with Compassion in Machine Learning (CaML). Per-file credits are in
[NOTICE](NOTICE).
