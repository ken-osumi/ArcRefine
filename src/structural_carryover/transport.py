"""Copy a configuration and its optional templates to a remote run."""

import json
from pathlib import Path

from structural_carryover.config import load_config


def pack(path: Path):
    config = load_config(path).to_dict()
    files = {}
    for i, target in enumerate(config['targets']):
        if target.get('template_pdb'):
            source = Path(target['template_pdb'])
            name = f'target_{i}{source.suffix}'
            files[name] = source.read_bytes()
            target['template_pdb'] = name
    return config, files


def unpack(config: dict, files: dict, folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        if Path(name).name != name:
            raise ValueError('input filenames must not contain directories')
        (folder/name).write_bytes(content)
    path = folder/'config.json'
    path.write_text(json.dumps(config))
    return load_config(path)
