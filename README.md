# MCMC-sampling-for-MHN
Here, the results of the manuscript [_Quantifying Uncertainty of Predictions from Cancer Progression Models_](docs/Quantifying_Uncertainty_of_Predictions_from_Cancer_Progression_Models.pdf) are reproduced.

## Contents

This repository contains:

- In [analysis](analysis) the scripts and utitlities to perform the analyses and produce the figures, including utilities.
- In [data](data) the patient mutation and metadata on which the analyses were performed.
- In [docs](docs) the manuscript.
- In [mcmc_sampling](mcmc_sampling) the script to run the sampling, including utilities beyond the [`mhn`](https://github.com/spang-lab/LearnMHN) package.
- In [results](results) the Maximum-Likelihood models, the posterior sampling results and the produced figures.

## Reproducing the Results

To reproduce the results in [Python](https://www.python.org/about/gettingstarted/), install the [`requirements.txt`](requirements.txt), ideally in a [virtual environment](https://docs.python.org/3/library/venv.html) by calling
```Bash
pip install -r requirements.txt
```
Then follow the READMEs in [results](results) to reproduce the sampling or in [analysis](analysis) to reproduce the analyses and figures.

## Contact

yanren-linda.hu@ur.de