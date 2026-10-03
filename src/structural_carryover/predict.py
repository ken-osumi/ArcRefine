"""Independent Boltz-2 prediction using Mosaic's native model interface."""

import argparse
import json
from pathlib import Path

from structural_carryover.config import Config, load_config, protein_sequence


def read_binder(path: Path) -> str:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if sum(line.startswith('>') for line in lines) != 1 or not lines[0].startswith('>'):
        raise ValueError('binder FASTA must contain exactly one record')
    return protein_sequence(''.join(line for line in lines[1:]), 'binder')


def predict(config: Config, binder: str, output: Path, *, recycling_steps=3, sampling_steps=25):
    import jax
    import numpy as np
    from mosaic.common import TOKENS
    from mosaic.models.boltz2 import Boltz2
    from mosaic.structure_prediction import TargetChain

    protein_sequence(binder, 'binder')
    if output.exists():
        raise FileExistsError(f'refusing to overwrite {output}')
    output.mkdir(parents=True)
    folder = Boltz2()
    chains = [TargetChain(sequence=binder, use_msa=False)] + [
        TargetChain(sequence=t.sequence, use_msa=t.use_msa) for t in config.targets
    ]
    features, writer = folder.target_only_features(chains)
    encoded = jax.nn.one_hot(np.array([TOKENS.index(a) for a in binder]), 20)
    prediction = folder.predict(
        PSSM=encoded, features=features, writer=writer,
        recycling_steps=recycling_steps, sampling_steps=sampling_steps,
        key=jax.random.key(config.seed),
    )
    jax.block_until_ready(prediction.pae)
    prediction.st.make_mmcif_document().write_file(str(output/'prediction.cif'))
    np.save(output/'pae.npy', np.asarray(prediction.pae))
    np.save(output/'plddt.npy', np.asarray(prediction.plddt))
    result = {
        'binder_sequence': binder,
        'binder_target_iptm': float(prediction.iptm),
        'chain_map': {'A': 'binder', **{chr(66+i): t.id for i,t in enumerate(config.targets)}},
    }
    (output/'confidence.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--binder', type=Path, required=True, help='single-record FASTA')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recycles', type=int, default=3)
    parser.add_argument('--sampling-steps', type=int, default=25)
    args = parser.parse_args()
    if args.recycles < 1 or args.sampling_steps < 1:
        parser.error('prediction steps must be positive')
    try:
        result = predict(load_config(args.config), read_binder(args.binder), args.output,
                         recycling_steps=args.recycles, sampling_steps=args.sampling_steps)
        print(json.dumps(result, indent=2))
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f'Input error: {exc}\n')


if __name__ == '__main__':
    main()
