import argparse
import numpy as np

import mhn.model
from mhn.mcmc import kernels
from mhn.mcmc.mcmc import MCMC
from mhn.optimizers import Penalty

from utils import l2_sym


parser = argparse.ArgumentParser(description="Run MCMC sampler")
parser.add_argument("--data_name", required=True,
                    help="Dataset name, either 'G13_LUAD_12 or "
                    "'G13_COAD_12'")
parser.add_argument(
    "--prior", help="Prior name, one of 'symsparse', 'l1', 'sym-l2'. "
    "Defaults to 'symsparse'")
parser.add_argument(
    "--kernel", help="Kernel name, one of 'MALA', 'RWM', 'smMALA'. "
    "Defaults to 'MALA'")

args = parser.parse_args()

data_name = args.data_name
prior = args.prior if args.prior else "symsparse"
kernel_name = args.kernel if args.kernel else "MALA"

kernel = {
    "MALA": kernels.MALAKernel,
    "RWM": kernels.RWMKernel,
    "smMALA": kernels.smMALAKernel
}[kernel_name]

penalty = {
    "symsparse": Penalty.SYM_SPARSE,
    "l1": Penalty.L1,
    "sym-l2": (l2_sym.gaussian_sym_log_prior,
               l2_sym.gaussian_sym_log_prior_grad,
               l2_sym.gaussian_sym_log_prior_hessian)
}[prior]

data = np.loadtxt(f"data/{data_name}.csv", delimiter=",", skiprows=1,
                  dtype=np.int32)
mhn_model = mhn.model.oMHN.load(
    f"results/mhns/{data_name}_{prior}_mle.csv")

mcmc_sampler = MCMC(
    mhn_model=mhn_model,
    data=data,
    penalty=penalty,
    kernel_class=kernel,
    thin=100,
)

mcmc_sampler.run()

np.save(
    f"results/mcmc/{data_name}_{prior}_{kernel_name}_"
    f"{str(mcmc_sampler.step_size).replace('.', '_')}.npy",
    mcmc_sampler.log_thetas)
