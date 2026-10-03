import json
from pathlib import Path

import pytest
from structural_carryover.config import load_config
from structural_carryover.inputs import validate_target_files

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("target", ["pdl1", "tnfa"])
def test_shipped_example_has_consistent_sequence_and_template(target):
    config = load_config(ROOT / "examples" / target / "config.json")
    validate_target_files(config)


@pytest.mark.parametrize(
    "update",
    [
        {"carryover": "false"},
        {"phase_steps": [1, 0, 1]},
        {"parent_sequence": "AAxA"},
        {"epitope_indices": [4]},
        {"epitope_indices": [0, 0]},
        {"samplng_steps": 25},
        {"targets": []},
        {"seed": -1},
    ],
)
def test_bad_inputs_fail_before_loading_models(tmp_path, update):
    raw = {"name": "test", "parent_sequence": "AAAA", "targets": [{"id": "T", "sequence": "AAAA"}]}
    raw.update(update)
    file = tmp_path / "config.json"
    file.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_config(file)



def test_arbitrary_target_and_template_transport(tmp_path):
    from structural_carryover.transport import pack, unpack

    path = tmp_path/'input.json'
    path.write_text(json.dumps({"name":"custom", "parent_sequence":"ACDE",
                               "targets":[{"id":"custom_protein", "sequence":"WYVTS", "use_msa":False}]}))
    data, files = pack(path)
    config = unpack(data, files, tmp_path/'remote')
    assert config.name == 'custom'
    assert config.targets[0].sequence == 'WYVTS'
    assert config.targets[0].use_msa is False
    assert not files


def test_mosaic_msa_default_and_boolean_validation(tmp_path):
    path=tmp_path/'input.json'
    data={"name":"custom", "parent_sequence":"ACDE", "targets":[{"id":"T", "sequence":"WYVTS"}]}
    path.write_text(json.dumps(data))
    assert load_config(path).targets[0].use_msa is True
    data['targets'][0]['use_msa']='false'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='use_msa'):
        load_config(path)


def test_standalone_prediction_accepts_one_sequence_only(tmp_path):
    from structural_carryover.predict import read_binder
    path=tmp_path/'binder.fasta'
    path.write_text('>binder\nACDE\nFGHI\n')
    assert read_binder(path)=='ACDEFGHI'
    path.write_text('>one\nACDE\n>two\nFGHI\n')
    with pytest.raises(ValueError, match='one record'):
        read_binder(path)
