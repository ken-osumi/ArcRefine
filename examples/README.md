# Example inputs

These examples illustrate the JSON interface; neither target is hardcoded in the
optimizer. For another protein, replace `parent_sequence` and the target chain
sequences, and supply optional template files and MSA settings.

| Folder | Target | Starting binder length |
| --- | --- | ---: |
| pdl1 | PD-L1 | 84 |
| pdl1_second | PD-L1 | 124 |
| tnfa | Trimeric TNFα | 116 |
| tnfa_second | Trimeric TNFα | 104 |

Each `source.json` records the origin of the example. MSAs use Mosaic’s standard service; set `use_msa` to false for query-only input.
`reference.pdb` defines the starting geometry for optional structural comparison;
its archived scaffold residue names can differ from the input sequence.
The JSON sequence is the optimizer input. Example data are CC BY 4.0.
