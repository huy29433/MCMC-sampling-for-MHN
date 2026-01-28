# Results

This directory contains the results of the scripts in `./mcmc_sampling`

The files are binary and can be opened in Python with
```python
import numpy as np

chains = np.load(f"G13_LUAD_12_sym-l2_1se_p_mle_00015625.npy")

print(type(chains))
>>> <class 'numpy.ndarray'>

print(chains.shape)
>>> (10, 44000, 156)
```

Above example contains the results of 10 chains with 44000 steps each, where each step consists of a flattened 13 x 12 oMHN parameter matrix.

TODO explain nomenclature