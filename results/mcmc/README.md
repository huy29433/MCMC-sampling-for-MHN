# Results

This directory contains the results of the scripts in `./mcmc_sampling`

The files are binary and can be opened in Python with
```python
import numpy as np

chains = np.load(f"G13_COAD_12_symsparse_MALA_0_001.npy")

print(type(chains))
>>> <class 'numpy.ndarray'>

print(chains.shape)
>>> (10, 1360, 156)
```

Above example contains the results of 10 chains with 1360 steps each, where each step consists of a flattened 13 x 12 oMHN parameter matrix.

The nomenclature is `<data_name>\_<penalty_name>\_<sampler_name>\_<step_size>.npy`.

The samples obtained using RWM or from the posterior of L1-trained MHNs

- _G13_COAD_12_l1_MALA_0_0015625.npy_
- _G13_LUAD_12_l1_MALA_0_001.npy_
- _G13_LUAD_12_sym-l2_RWM_0_09999999999999999.npy_

are too big to be uploaded to Github in their original form.
This is why we only uploaded every 100th of the (already thinned!) samples:

- _G13_COAD_12_l1_MALA_0_0015625_100th.npy_
- _G13_LUAD_12_l1_MALA_0_001_100th.npy_
- _G13_LUAD_12_sym-l2_RWM_0_09999999999999999_100th.npy_

Note that the analysis scripts were still applied to the original files, so in order to reproduce them for these configurations, please contact us or run the samplers yourself.