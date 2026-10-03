"""Evaluate saved parent/final fresh predictions against a starting complex.

Example: python scripts/evaluate.py runs/pdl1 --reference examples/pdl1/reference.pdb \
    --binder-chain B --target-chains A
ipSAE: DunbrackLab residue-wise d0 definition, PAE < 10 A, protein d0 floor 1 A.
Reference: https://github.com/DunbrackLab/IPSAE . No coordinate-distance cutoff
enters ipSAE; the 5 A heavy-atom cutoff applies only to contact recall.
"""
import argparse
import itertools
import json
from pathlib import Path

import gemmi
import numpy as np
from scipy.spatial import cKDTree


def residues(chain):
    return [r for r in chain if any(a.name == 'CA' for a in r)]


def ca(chain):
    return np.array([[r['CA'][0].pos.x, r['CA'][0].pos.y, r['CA'][0].pos.z]
                     for r in residues(chain)])


def fit(x, y):
    if x.shape != y.shape:
        raise ValueError('reference and predicted target lengths differ')
    u, _, vt = np.linalg.svd((x-x.mean(0)).T @ (y-y.mean(0)))
    d = np.eye(3)
    d[-1, -1] = np.linalg.det(u @ vt)
    r = u @ d @ vt
    t = y.mean(0)-x.mean(0) @ r
    return r, t, float(np.sqrt(np.mean(np.sum((x@r+t-y)**2, axis=1))))


def heavy(chains):
    xyz, idx, offset = [], [], 0
    for chain in chains:
        rs = residues(chain)
        for i, res in enumerate(rs):
            for atom in res:
                if not atom.element.is_hydrogen:
                    xyz.append([atom.pos.x, atom.pos.y, atom.pos.z])
                    idx.append(offset+i)
        offset += len(rs)
    return np.asarray(xyz), np.asarray(idx)


def contacts(binder, targets):
    bx, bi = heavy([binder])
    tx, ti = heavy(targets)
    neighbors = cKDTree(tx).query_ball_point(bx, 5.)
    return {(int(bi[i]), int(ti[j])) for i, js in enumerate(neighbors) for j in js}


def directional_ipsae(pae):
    valid = pae < 10.
    counts = valid.sum(axis=1)
    d0 = np.maximum(1., 1.24 * np.cbrt(np.maximum(counts, 26)-15.) - 1.8)
    terms = 1./(1.+(pae/d0[:, None])**2)
    scores = (terms*valid).sum(axis=1)/np.maximum(counts, 1)
    return float(scores.max(initial=0.))


def complex_ipsae(pae, lengths):
    cuts = np.cumsum([0, *lengths])
    if pae.shape != (cuts[-1], cuts[-1]) or not np.isfinite(pae).all():
        raise ValueError('invalid PAE array')
    pairs = {}
    for i in range(1, len(lengths)):
        forward = pae[:cuts[1], cuts[i]:cuts[i+1]]
        reverse = pae[cuts[i]:cuts[i+1], :cuts[1]]
        pairs[f'A-{chr(65+i)}'] = max(directional_ipsae(forward), directional_ipsae(reverse))
    return max(pairs.values()), pairs


def evaluate(folder, reference, binder_chain, target_chains):
    config = json.loads((folder/'config.json').read_text())
    result = json.loads((folder/'result.json').read_text())
    ref = gemmi.read_structure(str(reference))[0]
    rb, rt = ref[binder_chain], [ref[c] for c in target_chains]
    lengths = [len(config['parent_sequence']), *[len(t['sequence']) for t in config['targets']]]
    if len(rt) != len(lengths)-1:
        raise ValueError('target chain count mismatch')
    rc = contacts(rb, rt)
    if not rc:
        raise ValueError('reference has no binder-target contacts at 5 A')
    metrics = {}
    for prefix in ('parent', 'fresh'):
        pred = gemmi.read_structure(str(folder/f'{prefix}_prediction.cif'))[0]
        pb, pt = pred['A'], [pred[chr(66+i)] for i in range(len(rt))]
        sequence = ''.join(gemmi.find_tabulated_residue(x.name).one_letter_code for x in residues(pb))
        expected = config['parent_sequence'] if prefix == 'parent' else result['optimized_sequence']
        if sequence != expected:
            raise ValueError('saved structure sequence differs from expected sequence')
        # Only permute chains with identical target sequences, never unrelated entities.
        sequences = [t['sequence'] for t in config['targets']]
        poses = []
        for perm in itertools.permutations(range(len(rt))):
            if any(sequences[i] != sequences[j] for i, j in enumerate(perm)):
                continue
            r, t, target_rmsd = fit(np.concatenate([ca(pt[i]) for i in perm]),
                                    np.concatenate([ca(c) for c in rt]))
            pose = float(np.sqrt(np.mean(np.sum((ca(pb)@r+t-ca(rb))**2, axis=1))))
            poses.append((target_rmsd, pose, perm))
        best = min(x[0] for x in poses)
        mapping = min((x for x in poses if x[0] <= best+.5), key=lambda x: x[1])
        pc = contacts(pb, [pt[i] for i in mapping[2]])
        score, pair_scores = complex_ipsae(np.load(folder/f'{prefix}_pae.npy'), lengths)
        metrics[prefix] = {
            'ipsae': score, 'pair_ipsae': pair_scores,
            'contact_recall': len(rc & pc)/len(rc),
            'reference_contact_pairs': len(rc),
            'binder_ca_rmsd_A': fit(ca(pb), ca(rb))[2],
            'pose_ca_rmsd_A': mapping[1], 'target_chain_permutation': mapping[2],
        }
    delta = metrics['fresh']['ipsae']-metrics['parent']['ipsae']
    return {
        'name': config['name'], 'seed': config['seed'], 'carryover': config['carryover'],
        'steps': result['steps'], 'changed_fraction': 1-result['sequence_identity_to_parent'],
        **metrics, 'delta_ipsae': delta,
        'elapsed_seconds': result['elapsed_seconds'],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--binder-chain', required=True)
    parser.add_argument('--target-chains', nargs='+', required=True)
    args = parser.parse_args()
    report = evaluate(args.run, args.reference, args.binder_chain, args.target_chains)
    (args.run/'evaluation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
