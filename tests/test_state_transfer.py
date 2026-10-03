"""Exercise the real Mosaic optimizer on a small differentiable stateful loss."""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from mosaic.common import LossTerm, StateIndex
from structural_carryover.config import Config, Target
from structural_carryover.runner import optimize


class StateDependentLoss(LossTerm):
    state: jax.Array
    state_index: StateIndex | None

    def __call__(self, sequence, *, key):
        target = jax.nn.softmax(jnp.arange(sequence.shape[-1]) * (self.state + 1) / 5)
        value = jnp.square(sequence - target).sum()
        aux = {"state": self.state}
        if self.state_index is not None:
            return value, (aux, (self.state_index, self.state + 1))
        return value, aux

    def update_state(self, state):
        return eqx.tree_at(lambda m: m.state, self, state)


def test_carryover_updates_state_across_all_phases_and_off_does_not():
    config = Config(
        name="toy",
        parent_sequence="AAAA",
        targets=(Target("T", "AAAA"),),
        phase_steps=(2, 2, 2),
        seed=9,
    )
    start = jax.nn.softmax(jax.random.normal(jax.random.key(1), (4, 19)))
    on_loss = StateDependentLoss(jnp.array(0.0), StateIndex())
    off_loss = StateDependentLoss(jnp.array(0.0), None)
    on_steps, off_steps = [], []
    on_pssm, on_final = optimize(
        on_loss,
        initial_pssm=start,
        config=config,
        trajectory_fn=lambda *args: on_steps.append(args[0]),
    )
    off_pssm, off_final = optimize(
        off_loss,
        initial_pssm=start,
        config=replace(config, carryover=False),
        trajectory_fn=lambda *args: off_steps.append(args[0]),
    )
    assert float(on_final.state) == 6
    assert float(off_final.state) == 0
    assert on_steps == off_steps == ["soft", "soft", "sharpen", "sharpen", "final", "final"]
    assert np.isfinite(np.asarray(on_pssm)).all()
    np.testing.assert_allclose(np.asarray(on_pssm).sum(-1), 1.0, atol=1e-6)
    assert not np.allclose(on_pssm, off_pssm)

