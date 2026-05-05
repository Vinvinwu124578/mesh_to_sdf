# Overall tactile coverage summary

- files analyzed: 1071
- coverage definition: a surface point is counted as covered if its nearest tactile point distance is below a threshold.
- note: this is a point-cloud approximation of surface coverage, not an exact continuous surface-area proof.

- coverage@0.005: full=0, mean=0.374772, median=0.348949, min=0.032379, max=0.939540
- coverage@0.01: full=0, mean=0.492445, median=0.464485, min=0.054009, max=0.973434
- coverage@0.02: full=0, mean=0.564303, median=0.539562, min=0.082102, max=0.991306
- coverage@0.05: full=1, mean=0.730418, median=0.763621, min=0.170855, max=1.000000

## Category summary
- bottle: n=271, coverage@0.02 mean=0.633393, coverage@0.05 mean=0.791483, max_nn_dist mean=0.247153, max=0.711185
- bowl: n=177, coverage@0.02 mean=0.622166, coverage@0.05 mean=0.764354, max_nn_dist mean=0.229649, max=0.650813
- camera: n=112, coverage@0.02 mean=0.511132, coverage@0.05 mean=0.683600, max_nn_dist mean=0.277010, max=0.752807
- guitar: n=39, coverage@0.02 mean=0.845011, coverage@0.05 mean=0.962385, max_nn_dist mean=0.100470, max=0.164016
- jar: n=259, coverage@0.02 mean=0.528806, coverage@0.05 mean=0.712856, max_nn_dist mean=0.312062, max=0.770983
- mug: n=213, coverage@0.02 mean=0.448039, coverage@0.05 mean=0.628023, max_nn_dist mean=0.330081, max=0.708077

## Extremes
- best coverage@0.02: bowl / bowl__02880940__8d457deaf22394da65c5c31ac688ec4__models__model_normalized.npz / coverage=0.991306 / max_nn_dist=0.060032
- worst coverage@0.02: bowl / bowl__02880940__5019f979a6a360963a5e6305a3a7adee__models__model_normalized.npz / coverage=0.082102 / max_nn_dist=0.595175