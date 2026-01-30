import numpy as np
import pandas as pd
import mhn
import multiprocessing as mp


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
                n_samples: int = 1000, trajectory_num: int = 100):

    patients_unique = data.drop_duplicates()
    n_events = data.shape[1]

    probs_unique = np.zeros((patients_unique.shape[0], n_samples, n_events))

    log_theta_samples = log_thetas[np.random.choice(
        log_thetas.shape[0], n_samples)]

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(_sample_risks, [(
            log_theta, trajectory_num, patients_unique)
            for log_theta in log_theta_samples])

    for j, risks in enumerate(results):
        probs_unique[:, j, :] = risks

    probs = probs_unique[data.groupby(
        data.columns.tolist(), sort=False).ngroup()]

    return probs
