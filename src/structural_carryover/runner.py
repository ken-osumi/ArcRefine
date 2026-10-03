"""Mosaic binder optimization with structural carryover."""

import json
import time
from datetime import UTC, datetime
from pathlib import Path

from structural_carryover.config import Config
from structural_carryover.inputs import prepare_chains, validate_target_files


def phase_plan(length: int, steps: tuple[int, int, int]) -> list[dict]:
    """Three-phase APGM schedule; phase one hands off its best PSSM."""
    return [
        dict(
            name="soft",
            n_steps=steps[0],
            stepsize=0.2 * length**0.5,
            momentum=0.3,
            scale=1.0,
            logspace=False,
            take="best",
        ),
        dict(
            name="sharpen",
            n_steps=steps[1],
            stepsize=0.5 * length**0.5,
            momentum=0.0,
            scale=1.25,
            logspace=True,
            take="final",
        ),
        dict(
            name="final",
            n_steps=steps[2],
            stepsize=0.5 * length**0.5,
            momentum=0.0,
            scale=1.4,
            logspace=True,
            take="final",
        ),
    ]


def initial_probabilities(length: int, seed: int):
    """Initialize sequence probabilities with random logits."""
    import numpy as np

    rng = np.random.default_rng(seed)
    logits = rng.uniform(0.25, 0.75) * rng.gumbel(size=(length, 19))
    weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return (weights / weights.sum(axis=-1, keepdims=True)).astype(np.float32)


def optimize(loss, *, initial_pssm, config: Config, trajectory_fn):
    """Run the three optimization phases, carrying state when enabled."""
    import jax
    import jax.numpy as jnp
    from mosaic.optimizers import simplex_APGM

    pssm = initial_pssm
    for phase_number, phase in enumerate(
        phase_plan(len(config.parent_sequence), config.phase_steps)
    ):
        settings = dict(phase)
        phase_name = settings.pop("name")
        take = settings.pop("take")

        def callback(aux, updated_x, phase_name=phase_name):
            return trajectory_fn(phase_name, aux, updated_x)

        start = jnp.log(pssm + 1e-5) if settings["logspace"] else pssm
        result = simplex_APGM(
            loss_function=loss,
            x=start,
            key=jax.random.fold_in(jax.random.key(config.seed), phase_number + 1),
            max_gradient_norm=1.0,
            update_loss_state=config.carryover,
            trajectory_fn=callback,
            **settings,
        )
        pssm = result[1] if take == "best" else result[0]
        if config.carryover:
            loss = result[3]
    return pssm, loss


def build_loss(folder, mpnn, features, initial_state, config: Config):
    import jax.numpy as jnp
    import mosaic.losses.structure_prediction as sp
    from mosaic.common import TOKENS, StateIndex
    from mosaic.losses.boltz2 import MultiSampleBoltz2Loss
    from mosaic.losses.protein_mpnn import InverseFoldingSequenceRecovery
    from mosaic.losses.transformations import NoCys

    length = len(config.parent_sequence)
    bias = jnp.zeros((length, 20)).at[:, TOKENS.index("C")].set(-1e6)
    objective = (
        sp.BinderTargetContact()
        + sp.WithinBinderContact()
        + 10.0 * InverseFoldingSequenceRecovery(mpnn, temp=jnp.array(0.001), bias=bias)
        + 0.05 * sp.TargetBinderPAE()
        + 0.05 * sp.BinderTargetPAE()
        + 0.025 * sp.IPTMLoss()
        + 0.4 * sp.WithinBinderPAE()
        + 0.025 * sp.pTMEnergy()
        + 0.1 * sp.PLDDTLoss()
    )
    return NoCys(
        MultiSampleBoltz2Loss(
            joltz2=folder.model,
            features=features,
            loss=objective,
            deterministic=True,
            recycling_steps=config.recycling_steps,
            sampling_steps=config.sampling_steps,
            num_samples=config.num_samples,
            initial_recycling_state=initial_state if config.carryover else None,
            state_index=StateIndex() if config.carryover else None,
        )
    )


def run(config: Config, output: Path, *, predict: bool = True) -> dict:
    import jax
    import numpy as np
    from mosaic.common import TOKENS
    from mosaic.losses.boltz2 import Boltz2Output
    from mosaic.losses.transformations import NoCys
    from mosaic.models.boltz2 import Boltz2
    from mosaic.proteinmpnn import mpnn as mpnn_module
    from mosaic.structure_prediction import TargetChain

    validate_target_files(config)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite an existing output directory: {output}")
    if not any(d.platform == "gpu" for d in jax.devices()):
        raise ValueError("optimization requires a CUDA GPU; use --validate-only on a CPU")
    output.mkdir(parents=True)
    start = time.monotonic()
    (output / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    print("Loading Boltz-2 and soluble ProteinMPNN", flush=True)
    folder = Boltz2()
    mpnn = mpnn_module.load_mpnn_sol(0.05)
    chains = prepare_chains(config, output)
    length = len(config.parent_sequence)
    initial_state = None
    state_summary = {"carryover": config.carryover, "initialized": False}
    if config.carryover:
        print("Initializing single/pair representations from the parent sequence", flush=True)
        parent_features, _ = folder.target_only_features(
            [TargetChain(sequence=config.parent_sequence, use_msa=False), *chains]
        )
        parent_output = Boltz2Output(
            joltz2=folder.model,
            features=parent_features,
            recycling_steps=3,
            num_sampling_steps=25,
            key=jax.random.key(0),
            deterministic=True,
        )
        initial_state = jax.lax.stop_gradient(parent_output.trunk_state)
        jax.block_until_ready(initial_state)
        state_summary.update(
            initialized=True,
            single_shape=list(initial_state.s.shape),
            pair_shape=list(initial_state.z.shape),
        )
    features, _ = folder.binder_features(binder_length=length, chains=chains)
    loss = build_loss(folder, mpnn, features, initial_state, config)
    initial_pssm = jax.device_put(initial_probabilities(length, config.seed))
    np.save(output / "initial_sequence_probabilities.npy", np.asarray(initial_pssm))
    trajectory = []

    def record(phase_name, aux, updated_x):
        loss_value = float(aux["loss"])
        if not np.isfinite(loss_value):
            raise RuntimeError("optimization produced a non-finite loss; stopping")
        item = {
            "step": len(trajectory) + 1,
            "phase": phase_name,
            "loss_before_update": loss_value,
            "step_seconds": float(aux["time"]),
        }
        trajectory.append(item)
        with (output / "trajectory.jsonl").open("a") as stream:
            stream.write(json.dumps(item) + "\n")
        return item

    print(
        f"Optimizing {length} residues; carryover={config.carryover}; steps={config.phase_steps}",
        flush=True,
    )
    pssm, _ = optimize(loss, initial_pssm=initial_pssm, config=config, trajectory_fn=record)
    probabilities = NoCys.sequence(pssm)
    probabilities = np.asarray(jax.block_until_ready(probabilities))
    if not np.isfinite(probabilities).all():
        raise RuntimeError("non-finite final sequence probabilities")
    sequence = "".join(TOKENS[i] for i in probabilities.argmax(-1))
    np.save(output / "sequence_probabilities.npy", probabilities)
    (output / "optimized.fasta").write_text(
        f">{config.name}|carryover={config.carryover}|seed={config.seed}\n{sequence}\n"
    )
    result = {
        "name": config.name,
        "optimized_sequence": sequence,
        "sequence_identity_to_parent": sum(
            a == b for a, b in zip(sequence, config.parent_sequence, strict=True)
        )
        / length,
        "steps": len(trajectory),
        "carryover_state": state_summary,
        "fresh_prediction": False,
    }
    if predict:
        print(
            "Predicting the discrete final sequence afresh, without carried state or templates",
            flush=True,
        )
        fresh_targets = [
            TargetChain(sequence=t.sequence, use_msa=t.use_msa) for t in chains
        ]
        for prefix, evaluated_sequence in (("parent", config.parent_sequence), ("fresh", sequence)):
            fresh_features, writer = folder.target_only_features(
                [TargetChain(sequence=evaluated_sequence, use_msa=False), *fresh_targets]
            )
            # Supplying the discrete binder encoding also tells Mosaic where the
            # binder ends when it constructs the binder-versus-target ipTM mask.
            discrete_binder = jax.nn.one_hot(np.array([TOKENS.index(a) for a in evaluated_sequence]), 20)
            prediction = folder.predict(
                PSSM=discrete_binder,
                features=fresh_features,
                writer=writer,
                recycling_steps=3,
                sampling_steps=25,
                key=jax.random.fold_in(jax.random.key(config.seed), 100),
            )
            jax.block_until_ready(prediction.pae)
            prediction.st.make_mmcif_document().write_file(str(output / f"{prefix}_prediction.cif"))
            np.save(output / f"{prefix}_pae.npy", np.asarray(prediction.pae))
            np.save(output / f"{prefix}_plddt.npy", np.asarray(prediction.plddt))
            result[f"{prefix}_binder_target_iptm"] = float(prediction.iptm)
        result["fresh_prediction"] = True
        result["prediction_chain_map"] = {
            "A": "binder",
            **{chr(66 + i): t.id for i, t in enumerate(config.targets)},
        }
    result["elapsed_seconds"] = time.monotonic() - start
    result["completed_at"] = datetime.now(UTC).isoformat()
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
