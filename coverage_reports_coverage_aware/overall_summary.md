# Overall coverage-aware tactile coverage summary

- unique files analyzed: 1853
- de-duplication rule: identical synset/object paths were counted once, so duplicated camera files were not double-counted.
- coverage definition: a surface point is counted as covered if its nearest tactile point distance is below a threshold.
- note: this is a point-cloud approximation of surface coverage, not an exact continuous surface-area proof.

- coverage@0.005: full=0, mean=0.488318, median=0.461451, min=0.051889, max=0.954038
- coverage@0.01: full=0, mean=0.637819, median=0.635987, min=0.102374, max=0.993068
- coverage@0.02: full=0, mean=0.719639, median=0.730034, min=0.140668, max=0.999685
- coverage@0.05: full=79, mean=0.854371, median=0.904140, min=0.281311, max=1.000000

## Category summary
- bathtub: n=272, coverage@0.02 mean=0.723615, coverage@0.05 mean=0.857912, max_nn_dist mean=0.184002, max=0.787351
- bottle: n=273, coverage@0.02 mean=0.799125, coverage@0.05 mean=0.923033, max_nn_dist mean=0.192282, max=0.973158
- bowl: n=178, coverage@0.02 mean=0.706929, coverage@0.05 mean=0.839379, max_nn_dist mean=0.176656, max=1.221214
- camera: n=113, coverage@0.02 mean=0.645807, coverage@0.05 mean=0.793934, max_nn_dist mean=0.283099, max=1.162061
- chair: n=272, coverage@0.02 mean=0.670068, coverage@0.05 mean=0.811539, max_nn_dist mean=0.316777, max=0.908573
- guitar: n=259, coverage@0.02 mean=0.930804, coverage@0.05 mean=0.994548, max_nn_dist mean=0.064236, max=0.182134
- jar: n=272, coverage@0.02 mean=0.659285, coverage@0.05 mean=0.829515, max_nn_dist mean=0.265845, max=1.096736
- mug: n=214, coverage@0.02 mean=0.546894, coverage@0.05 mean=0.723041, max_nn_dist mean=0.325605, max=0.653639

## Extremes
- best coverage@0.02: bowl / bowl__02880940__d2e1dc9ee02834c71621c7edb823fc53__models__model_normalized.npz / coverage=0.999685 / max_nn_dist=0.026621
- worst coverage@0.02: chair / chair__03001627__1767c5e3771b0510f5225bf5a419e95__models__model_normalized.npz / coverage=0.140668 / max_nn_dist=0.532000