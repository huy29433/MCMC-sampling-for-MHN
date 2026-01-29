import numpy as np

n = 12

def gaussian_sym_log_prior(log_theta: np.ndarray) -> float:
    """Computes a gaussian version of the sym_sparse prior density.

    Args:
        log_theta (np.ndarray): Value at which to evaluate the prior

    Returns:
        float: lam * (sum(theta_ij^2 + theta_ji^2 - theta_ij*theta_ji)
        + sum(theta_ii^2)
        (+ sum(omega_i^2)))
    """
    _log_theta = log_theta.copy().reshape(n + 1, n)

    log_theta_no_diag = _log_theta[:n].copy()

    np.fill_diagonal(log_theta_no_diag, 0)

    _log_theta_squared = log_theta_no_diag**2

    # halve the sum to avoid double counting
    prior = 0.5 * np.sum(
        _log_theta_squared
        + _log_theta_squared.T
        - log_theta_no_diag * log_theta_no_diag.T
    ) + np.sum(
        _log_theta[np.diag_indices(n)] ** 2
    )

    prior += np.sum(_log_theta[-1] ** 2)

    return prior


def gaussian_sym_log_prior_grad(
    log_theta: np.array,
) -> np.ndarray:
    """Computes the gradient of the symmetric l2 prior

    Args:
        log_theta (np.array): Value at which to evaluate the
        gradient of the log prior

    Returns:
        np.ndarray: Gradient of the log prior density
    """

    _log_theta = log_theta.reshape(n + 1, n)
    grad = 2 * _log_theta
    grad[:n] -= _log_theta[:n].T
    grad[np.diag_indices(n)] = 2 * \
        _log_theta[np.diag_indices(n)]

    return grad


hessian = np.eye(n**2 + n) * 2

# get all off-diagonal indices
theta_off_diag_indices = np.triu_indices(n, k=1)
hessian_row_indices = np.ravel_multi_index(
    theta_off_diag_indices, (n, n))
hessian_col_indices = np.ravel_multi_index(
    theta_off_diag_indices[::-1], (n, n))
hessian[hessian_row_indices, hessian_col_indices] = -1
hessian[hessian_col_indices, hessian_row_indices] = -1


def gaussian_sym_log_prior_hessian(
    log_theta: np.ndarray,
) -> np.ndarray:
    """Computes the Hessian of the gaussian sym_sparse log prior density.
    Args:
        log_theta (np.ndarray): Value at which to evaluate the Hessian of the log prior
    Returns:
        np.ndarray: Hessian of the log prior density
    """

    return hessian
