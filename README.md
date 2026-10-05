# BindRelay

Refine an existing protein binder design while preserving its fold and binding
mode. BindRelay uses structural carryover: Boltz-2's single and pair
representations pass between sequence updates. The resulting sequence is then
predicted independently with Boltz-2.

Accompanies [BindRelay refines de novo binder designs while preserving binding modes](https://foldarc.com/research/structural-carryover/paper/).

[Manuscript V18 · Zenodo version 2.2](https://doi.org/10.5281/zenodo.23153943) ·
[Research overview](https://foldarc.com/research/structural-carryover/)

Built on [Mosaic](https://github.com/escalante-bio/mosaic), with structural
carryover added to its optimization loop. The `structural-carryover` command
names are retained for compatibility.

## Install

Use Linux with an NVIDIA CUDA GPU, Git, `libgomp1` and
[uv](https://docs.astral.sh/uv/getting-started/installation/).

```sh
git clone https://github.com/ken-osumi/BindRelay.git
cd BindRelay
uv venv --python 3.12
. .venv/bin/activate
uv pip sync requirements-linux-cuda12.lock
uv pip install --no-deps .
```

Boltz-2 parameters download on first use. Soluble ProteinMPNN weights are included.
The package includes the required Mosaic modules; use a separate environment
from other Mosaic installations. GPU memory needs grow with complex size and the number of samples.

## Optimize your own binder

Create `design.json` with your starting binder sequence and one entry per target
chain. Replace the illustrative sequences below with your own proteins.

```json
{
  "name": "my_design",
  "parent_sequence": "ACDEFGHIKLMNPQRSTVWY",
  "targets": [{"id": "T", "sequence": "MNPQRSTVWYACDEFGHIKL"}]
}
```

```sh
structural-carryover design.json --validate-only
structural-carryover design.json --output run
```

Target chains can optionally specify `template_pdb` with `template_chain`.
Paths are relative to the JSON file. MSA handling is Mosaic's standard behavior:
`use_msa: true` (default) requests an MSA through its service; `false` uses only
the query sequence. The binder uses query-only features.

The starting complex prediction initializes the carried state. Sequence
probabilities start randomly and are optimized using Mosaic's structural losses
and soluble ProteinMPNN. Defaults use 50/50/15 updates; change them with
`--steps`, or disable carryover with `--carryover off`.
[Method details](docs/METHOD.md) describe the objective and settings.

`examples/pdl1/config.json` and `examples/tnfa/config.json` are runnable examples.
The optimizer accepts other protein targets through the same JSON interface.

## Independent prediction

Optimization also predicts the parent and final sequences separately, without
carried state or target templates. Use `--no-predict` to skip this step.
To predict a sequence separately, including one produced by another method:

```sh
structural-carryover-predict design.json --binder run/optimized.fasta --output prediction
```

The standalone predictor uses the target sequences and MSA settings from
`design.json`, with three recycles and 25 diffusion steps by default. It does not
load an optimization trajectory or ProteinMPNN.

## Outputs

Optimization writes `optimized.fasta`, sequence probabilities, loss history and
settings. The separate parent/final predictions are `parent_prediction.cif` and
`fresh_prediction.cif`, with corresponding PAE and pLDDT arrays. The standalone
predictor writes `prediction.cif`, `pae.npy`, `plddt.npy` and `confidence.json`.
CIF B-factors are placeholders; use the saved pLDDT arrays (0–1 scale).

An optional script reports contact retention, structural displacement and ipSAE
against your starting reference complex:

```sh
python scripts/evaluate.py run --reference starting_complex.pdb --binder-chain B --target-chains A
```

It reports measurements without a pass/fail rule. Choose criteria appropriate to
your use. Structural confidence alone does not establish experimental binding.

## Optional: run on Modal

With the Modal CLI installed and your account authenticated:

```sh
modal run modal_run.py --config design.json --output-dir run
modal run modal_run.py --config design.json --mode predict --binder run/optimized.fasta --output-dir prediction
```

The configuration and optional template files are uploaded. Runs use one H200
GPU on your account; charges apply. `--smoke` uses one update per phase for an
installation check. The launcher has a 40-minute timeout.

## License

FoldArc's code and modifications: **[PolyForm Noncommercial 1.0.0](LICENSE)**.
Noncommercial use is permitted under its terms; commercial use requires
permission from **ken@foldarc.com**. Universities and public research institutions
may use this software under this license without obtaining a separate
[patent license](LICENSE#L51) from FoldArc.
Third-party components retain their original licenses; see
[license scope](LICENSE_SCOPE.md) and
[attribution](THIRD_PARTY_NOTICES.md). Citation metadata is in `CITATION.cff`.
