import numpy as np
import pandas as pd
import mhn
import multiprocessing as mp
from typing import Literal


def _sample_risks(log_theta, trajectory_num: int, data: pd.DataFrame):

    occurences = np.zeros((data.shape[0], 12), dtype=int)

    model = mhn.model.oMHN(log_theta.reshape(13, 12))

    for i, (_, state) in enumerate(data.iterrows()):

        for sample in model.sample_trajectories(
            trajectory_num=trajectory_num,
            initial_state=state.to_numpy(),
            timed=1,
        )[0]:
            occurences[i, sample] += 1

    risks = occurences / trajectory_num

    return risks


def event_risks(log_thetas: np.ndarray, data: pd.DataFrame,
                n_samples: int | Literal["all"] = 1000,
                trajectory_num: int = 100):

    patients_unique = data.drop_duplicates()
    n_events = data.shape[1]

    if n_samples == "all":
        n_samples = log_thetas.shape[0]
        log_theta_samples = log_thetas
    else:
        log_theta_samples = log_thetas[np.random.choice(
            log_thetas.shape[0], n_samples)]

    probs_unique = np.zeros((patients_unique.shape[0], n_samples, n_events))

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(_sample_risks, [(
            log_theta, trajectory_num, patients_unique)
            for log_theta in log_theta_samples])

    for j, risks in enumerate(results):
        probs_unique[:, j, :] = risks

    probs = probs_unique[data.groupby(
        data.columns.tolist(), sort=False).ngroup()]

    return probs


def _sample_positions(log_theta, trajectory_num: int, n_bins: int):

    model = mhn.model.oMHN(log_theta.reshape(13, 12))

    position_counts = np.zeros((12, n_bins))
    bin_range = np.arange(n_bins)

    for trajectory in model.sample_trajectories(
            trajectory_num=trajectory_num)[0]:
        _len = len(trajectory)
        if _len == 0:
            continue
        position_counts[np.repeat(
            trajectory, (n_bins - 1) // _len + 1)[:n_bins], bin_range] += 1

    return position_counts


def event_positions(log_thetas: np.ndarray,
                    n_samples: int | Literal["all"] = 100,
                    trajectory_num: int = 50_000, n_bins: int = 100):

    if n_samples == "all":
        n_samples = log_thetas.shape[0]
        log_theta_samples = log_thetas
    else:
        log_theta_samples = log_thetas[np.random.choice(
            log_thetas.shape[0], n_samples)]

    positions = np.empty((n_samples, 12, n_bins))

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(_sample_positions, [(
            log_theta, trajectory_num, n_bins)
            for log_theta in log_theta_samples])

    for j, position_counts in enumerate(results):
        positions[j, :, :] = position_counts

    return positions
