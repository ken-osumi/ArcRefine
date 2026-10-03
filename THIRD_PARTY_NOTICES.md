# Third-party notices

This work builds on Mosaic by Nick Boyd and Escalante Bio, Boltz-2 by the Boltz
team, Joltz by Nick Boyd, and ProteinMPNN/soluble ProteinMPNN by their authors.
These projects provide the differentiable models, loss terms, optimizer,
structure handling, and inverse-folding model on which structural carryover runs.

| Component | Included or installed | License |
| --- | --- | --- |
| Mosaic upstream portions | Selected source files under `src/mosaic` | MIT; `licenses/Mosaic-MIT.txt` |
| FoldArc Mosaic modifications | Two source files identified in `docs/vendor-provenance.json` | PolyForm Noncommercial; upstream portions remain MIT |
| ProteinMPNN PyTorch implementation and soluble checkpoint | Under `src/mosaic/proteinmpnn` | MIT; `licenses/ProteinMPNN-MIT.txt` and Mosaic's notice |
| Joltz | Installed from pinned upstream Git revision | MIT; `licenses/Joltz-MIT.txt` |
| Boltz | Installed dependency; model parameters downloaded separately | See upstream terms; code MIT notice in `licenses/Boltz-MIT.txt` |
| Other Python dependencies | Installed separately; see the dependency lockfile | Their respective upstream licenses |

Mosaic upstream base: `bca3152befde5a4846430234deefe7946486e6e8` in
<https://github.com/escalante-bio/mosaic>.

FoldArc's modifications and source checksums are documented in `docs/`.

Joltz source: <https://github.com/nboyd/joltz> at
`ed0f04257dac85bd4b7bf45521cc280f86d38ded`.

ProteinMPNN source: <https://github.com/dauparas/ProteinMPNN>.
Soluble checkpoint: `soluble_v_48_020.pt`, SHA256
`7af52d090172c230c7f0e9d21e02203f6b3a38b16db58d3c7a3960e0a9a6e31a`.

Please cite the underlying models as well as the structural-carryover paper
when reporting work that uses them.

The pinned Boltz package uses the Escalante Bio dependency-compatibility fork
of <https://github.com/jwohlwend/boltz>, at
<https://github.com/escalante-bio/boltz/tree/1acc397b6e81f30dc07a80d17a26c38f9c6f6942>.
Installation requires access to this revision and the pinned Joltz revision.
Neither dependency's complete source archive is mirrored in this release.
