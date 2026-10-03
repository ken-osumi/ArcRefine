"""Validate target templates and construct native Mosaic inputs."""

from pathlib import Path

import gemmi

from structural_carryover.config import Config, Target


def template_chain(target: Target):
    if target.template_pdb is None:
        return None
    structure = gemmi.read_structure(target.template_pdb)
    if len(structure) != 1:
        raise ValueError(f"{target.id}: template must have exactly one coordinate model")
    structure.remove_alternative_conformations()
    chains = [chain for chain in structure[0] if chain.name == target.template_chain]
    if len(chains) != 1:
        raise ValueError(f"{target.id}: template chain must occur exactly once")
    chain = chains[0].clone()
    # Only canonical protein residues are admitted; do not silently remove gaps or ligands.
    sequence = gemmi.one_letter_code([res.name for res in chain])
    if sequence != target.sequence:
        raise ValueError(f"{target.id}: template residue sequence differs from configured sequence")
    if any(not res.find_atom("CA", "*") for res in chain):
        raise ValueError(f"{target.id}: template has residues without C-alpha atoms")
    return chain



def validate_target_files(config: Config) -> None:
    for target in config.targets:
        template_chain(target)


def prepare_chains(config: Config, output: Path | None = None):
    from mosaic.structure_prediction import TargetChain

    return [TargetChain(sequence=t.sequence, template_chain=template_chain(t),
                        use_msa=t.use_msa) for t in config.targets]
