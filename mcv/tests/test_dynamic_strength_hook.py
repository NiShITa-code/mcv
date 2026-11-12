"""Dynamic strength: a changed coefficient really reaches the hook (arithmetic: test_scheduled_coef.py)."""
import pytest

torch = pytest.importorskip("torch")

from mcv.config import ScheduleSpec, SteeringSpec
from mcv.tests._paths import requires_upstream

pytestmark = requires_upstream()  # ActivationSteerer and the train fixtures come from upstream


def _spec(coef=-1.5, schedule=None):
    return SteeringSpec(vector_path="v.pt", layer=26, coef=coef, schedule=schedule)


def test_changing_coeff_changes_the_hooked_output():
    from activation_steer import ActivationSteerer
    layer = torch.nn.Linear(4, 4, bias=False)
    torch.nn.init.zeros_(layer.weight)

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("C", (), {"hidden_size": 4})()
            self.model = type("L", (), {})()
            self.model.layers = torch.nn.ModuleList([layer])
            self.p = torch.nn.Parameter(torch.zeros(1))
        def forward(self, x):
            return self.model.layers[0](x)

    m = M()
    with ActivationSteerer(m, torch.ones(4), coeff=1.0, layer_idx=0, positions="all") as steerer:
        x = torch.zeros(1, 2, 4)
        assert torch.allclose(m(x), torch.ones(1, 2, 4))
        steerer.coeff = 3.0
        assert torch.allclose(m(x), 3 * torch.ones(1, 2, 4))


from mcv.tests.test_train import HIDDEN, N_LAYERS, _cfg, tiny_qwen, tiny_tokenizer, workspace  # noqa: E402,F401


def test_schedule_reaches_the_hook_on_every_training_step_including_after_a_checkpoint(workspace, monkeypatch):
    """End to end on the tiny model. A checkpoint exits the hook and re-enters a NEW steerer, so a
    schedule that only updated the first steerer would freeze at the starting strength after it."""
    import activation_steer
    from mcv import train
    from mcv.config import ScheduleSpec, SteeringSpec
    from mcv.vectors import VectorMetadata, write_sidecar

    tmp_path, training_file = workspace
    vec_path = tmp_path / "v.pt"
    stack = torch.ones(N_LAYERS + 1, HIDDEN)
    torch.save(stack, vec_path)
    write_sidecar(vec_path, VectorMetadata(
        trait="t", vector_type="response_avg_diff", source_model="tiny/qwen2", hidden_size=HIDDEN,
        n_layers=N_LAYERS, shape=list(stack.shape), norms_by_layer=[float(n) for n in stack.float().norm(dim=-1)]))

    seen = []
    original = activation_steer.ActivationSteerer._hook_fn

    def spy(self, module, ins, out):
        seen.append(self.coeff)
        return original(self, module, ins, out)

    monkeypatch.setattr(activation_steer.ActivationSteerer, "_hook_fn", spy)
    spec = SteeringSpec(vector_path=str(vec_path), layer=2, coef=1.0, positions="all",
                        schedule=ScheduleSpec(start_fraction=0.0, end_multiple=2.0))
    train.run(_cfg(tmp_path, training_file, steering=spec, checkpoint_fractions=[0.5]), repo_root=tmp_path)

    # 4 optimizer steps on this fixture (7 train rows, batch 2). The hook also fires on the checkpoint
    # evaluation and once more in the end-of-run held-out loss, so extra repeated values appear.
    # A frozen schedule (a stale first steerer after the checkpoint) would never reach 1.75 or 2.0.
    assert seen == sorted(seen), seen
    assert {round(c, 2) for c in seen} >= {1.25, 1.5, 1.75, 2.0}, seen
