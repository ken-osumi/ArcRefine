"""Validated, portable inputs. Relative file paths resolve beside the JSON file."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


def protein_sequence(value: str, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty protein sequence")
    if set(value) - AMINO_ACIDS:
        raise ValueError(f"{label} must contain only uppercase canonical amino acids")
    return value


@dataclass(frozen=True)
class Target:
    id: str
    sequence: str
    template_pdb: str | None = None
    template_chain: str | None = None
    use_msa: bool = True


@dataclass(frozen=True)
class Config:
    name: str
    parent_sequence: str
    targets: tuple[Target, ...]
    carryover: bool = True
    phase_steps: tuple[int, int, int] = (50, 50, 15)
    seed: int = 0
    recycling_steps: int = 1
    sampling_steps: int = 25
    num_samples: int = 4

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: Path) -> Config:
    path = path.resolve()
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a JSON object")
    unknown = set(raw) - set(Config.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown configuration fields: {sorted(unknown)}")
    for key in ("name", "parent_sequence", "targets"):
        if key not in raw:
            raise ValueError(f"missing configuration field: {key}")
    protein_sequence(raw["parent_sequence"], "parent_sequence")
    if not isinstance(raw["name"], str) or not raw["name"].strip():
        raise ValueError("name must be a nonempty string")
    if not isinstance(raw["targets"], list) or not 1 <= len(raw["targets"]) <= 25:
        raise ValueError("targets must list between 1 and 25 protein chains")
    targets = []
    for i, item in enumerate(raw["targets"]):
        if not isinstance(item, dict):
            raise ValueError(f"target {i} must be an object")
        extra = set(item) - set(Target.__dataclass_fields__)
        if extra:
            raise ValueError(f"unknown target fields: {sorted(extra)}")
        if "id" not in item or "sequence" not in item:
            raise ValueError(f"target {i} needs id and sequence")
        item = dict(item)
        protein_sequence(item["sequence"], f"target {i}")
        if not isinstance(item["id"], str) or not item["id"]:
            raise ValueError("target id must be a nonempty string")
        if bool(item.get("template_pdb")) != bool(item.get("template_chain")):
            raise ValueError("template_pdb and template_chain must be supplied together")
        for key in ("template_pdb",):
            if item.get(key) is not None:
                resolved = (path.parent / item[key]).resolve()
                if not resolved.is_file():
                    raise ValueError(f"missing {key}: {resolved}")
                item[key] = str(resolved)
        if type(item.get("use_msa", True)) is not bool:
            raise ValueError("use_msa must be true or false")
        targets.append(Target(**item))
    if len({t.id for t in targets}) != len(targets):
        raise ValueError("target ids must be unique")
    raw["targets"] = tuple(targets)
    if "phase_steps" in raw:
        steps = raw["phase_steps"]
        if not isinstance(steps, list) or len(steps) != 3:
            raise ValueError("phase_steps must contain three positive integers")
        raw["phase_steps"] = tuple(steps)
    config = Config(**raw)
    if type(config.carryover) is not bool:
        raise ValueError("carryover must be true or false")
    if type(config.seed) is not int or not 0 <= config.seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    values = {
        "phase_steps": config.phase_steps,
        "recycling_steps": (config.recycling_steps,),
        "sampling_steps": (config.sampling_steps,),
        "num_samples": (config.num_samples,),
    }
    for name, numbers in values.items():
        if any(type(n) is not int or n < 1 for n in numbers):
            raise ValueError(f"{name} must contain positive integers")
    return config
