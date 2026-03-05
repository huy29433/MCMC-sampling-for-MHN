# Analysis

This directory contains Jupyter notebooks to reproduce all results and figures from the paper.

## Contents

- [_algorithm_comparison.ipynb_](algorithm_comparison.ipynb) contains the performance comparison of RWM, MALA and smMALA (fig. 2).
- [_event_positions.ipynb_](event_positions.ipynb) contains estimation of temporal event positions (fig.s 3B, 4B).
- [_example_parameter_matrix.ipynb_](example_parameter_matrix.ipynb) contains estimation of temporal event positions (fig. 1).
- [_mutation_risk.ipynb_](mutation_risk.ipynb) contains estimation of mutation risks and survival (fig.s 3C, 3D, 4C, 4D, and supplementary fig.s SF1, SF2).
- [_param_disk.ipynb_](param_disk.ipynb) contains the plotting of the paramter matrices and principal component analyses (fig.s 3A, 4A, and supplementary fig.s SF3-SF6).
- [_sampler_metric.ipynb_](param_disk.ipynb) contains the calculation of the MCMC metrics (table 1).

## Important Notes

### Sample Subsets

Note that the samples obtained using RWM or from the posterior of L1-trained MHNs are too big to be stored in this repository.
Therefore only a subset of each file is stored here.
- In order to reproduce the exact results for these configurations, contact us for the original files or run the samplers yourself.
- For an approximate reproduction of the results use the uploaded samples, adapting the code in the analysis scripts to the reduced size if needed.

### Runtimes

Risks are estimated by creating a large number of trajectories from the MCMC samples.
This can take some time (up to 10 min).
Reducing the number of trajectories reduces this runtime but will not reproduce the results accurately.