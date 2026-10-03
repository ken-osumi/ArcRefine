# Method

The optimizer combines Mosaic's differentiable Boltz-2 and soluble ProteinMPNN
objectives with transfer of single and pair representations between sequence updates.

## Structural carryover

A prediction of the starting binder and target initializes the single and pair
representations. After each gradient evaluation, the updated representations
are retained with gradients stopped and supplied to the next sequence update.
`--carryover off` disables this initialization and transfer. Ordinary recycling
within each prediction is unchanged.

Targets are supplied as sequences with optional templates and Mosaic's MSA
settings. The binder sequence probabilities start from random logits; binder
coordinates are not inputs.

The Mosaic changes are limited to returning the updated loss state between
optimization phases and transferring the Boltz-2 representations.
[The patch](foldarc-mosaic.patch) records the complete diff. MSA loading and
attention use upstream Mosaic unchanged.

## Default objective and schedule

The three APGM phases use 50, 50 and 15 updates. Step sizes are 0.2, 0.5 and 0.5
times the square root of binder length; momenta are 0.3, 0 and 0; scale factors
are 1, 1.25 and 1.4. Phase 1 passes its best sequence probabilities to phase 2;
phases 2 and 3 pass their final probabilities. Representations carry forward
from the last gradient evaluation.

The loss combines binder-target and within-binder contacts (weight 1 each),
soluble ProteinMPNN sequence recovery (10), directional interchain PAE (0.05
each), within-binder PAE (0.4), ipTM (0.025), pTM (0.025) and pLDDT (0.1).
ProteinMPNN uses coordinate noise 0.05, temperature 0.001, 16 samples and
10 Jacobi iterations. Cysteine is excluded from the optimized sequence.
Optimization defaults to 1 recycle, 25 diffusion steps and 4 samples per
gradient evaluation, configurable through the JSON input.

Phase-1 loss is evaluated at the momentum extrapolation point. Mosaic's
best-PSSM selection associates that loss with the subsequent updated PSSM.

## Independent prediction

The parent and final discrete sequences are predicted with 3 recycles and 25
diffusion steps, without carried representations or target templates, using
the configured target MSA settings. `--no-predict` skips this step. The standalone
prediction command runs the same model without loading the optimization loop.
