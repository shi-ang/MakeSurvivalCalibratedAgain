import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils.util_survival import xcal_from_hist

from SurvivalEVAL.Evaluations.D_Calibration import d_calibration, create_censor_binning
from SurvivalEVAL.Evaluations.util import check_and_convert, NumericArrayLike


def cond_xcal(
        X: NumericArrayLike,
        event_indicators: NumericArrayLike,
        predict_probs: NumericArrayLike,
        conditions: list,
        num_bins: int = 10,
        delta: float = 0.1
) -> float:
    """
    Compute the conditional D-Calibration score for a given condition.
    :param X: NumericArrayLike of shape [n_samples, n_features], the feature matrix
    :param event_indicators: NumericArrayLike of shape [n_samples, n_time_points], the binary event indicators
    :param predict_probs: NumericArrayLike of shape [n_samples, n_time_points], the predicted probabilities
    :param conditions: list of function that takes in a feature vector and returns a condition
    :param num_bins: number of bins to use for calibration
    :param delta: minimum fraction of samples in a bin
    :return: dcal: float, dcal_scores: numpy array of shape [n_time_points]
    """
    X = check_and_convert(X)
    event_indicators, predict_probs = check_and_convert(event_indicators, predict_probs)

    if conditions:
        cond_xcals = []
        for condition in conditions:
            category_map = condition(X)
            idx = (category_map == 1)

            # Check if the category has enough samples
            if idx.mean() < delta:
                continue
            else:
                _, dcal_hist = d_calibration(predict_probs[idx], event_indicators[idx], num_bins)
                cond_xcals.append(xcal_from_hist(dcal_hist))
        return max(cond_xcals)
    else:
        # If no conditions are provided, return the None
        return None


def wsc_xcal(
        X: NumericArrayLike,
        event_indicators: NumericArrayLike,
        predict_probs: NumericArrayLike,
        num_bins: int = 10,
        delta: float = 0.33,
        test_size=0.75,
        M=1000,
        random_state: int = 42,
        verbose: bool = False,
) -> float:
    """
    Calculate the worst-slab (D-)calibration score for a given condition.
    :param X: NumericArrayLike of shape [n_samples, n_features], the feature matrix
    :param event_indicators: NumericArrayLike of shape [n_samples, ], the binary event indicators
    :param predict_probs: NumericArrayLike of shape [n_samples, ], the predicted probabilities
    :param num_bins: int, the number of bins to use for calibration
    :param delta: float, the minimum fraction of samples in a slab
    :param test_size: float, the fraction of samples to use for testing
    :param M: int, the number of random directions to sample
    :param random_state: int, the random seed
    :param verbose: bool, whether to print progress
    :return: float, the worst-slab calibration score
    """
    X = check_and_convert(X)
    event_indicators, predict_probs = check_and_convert(event_indicators, predict_probs)

    def wsc_vab(x, event_indicators, predict_probs, num_bins, v, a, b):
        z = np.dot(x, v)
        idx = np.where((z >= a) * (z <= b))
        _, dcal_hist = d_calibration(predict_probs[idx], event_indicators[idx], num_bins)
        return xcal_from_hist(dcal_hist)

    X_train, X_test, e_train, e_test, pred_train, pred_test = train_test_split(
        X, event_indicators, predict_probs, test_size=test_size, random_state=random_state)
    # Find adversarial parameters that maximize the calibration score in the 'train' set
    wsc_star, v_star, a_star, b_star = worst_slab(X_train, e_train, pred_train, num_bins,
                                                  delta=delta, M=M, verbose=verbose)
    # Estimate worst-slab calibration score in the 'test' set
    xcal = wsc_vab(X_test, e_test, pred_test, num_bins, v_star, a_star, b_star)
    return xcal


def worst_slab(X, event_indicators, predict_probs, num_bins, delta=0.33, M=1000, verbose=False):
    """ Find adversarial parameters that maximize the calibration score in the 'train' set."""
    V = sample_sphere(M, p=X.shape[1])
    wsc_list = [[]] * M
    a_list = [[]] * M
    b_list = [[]] * M
    for m in tqdm(range(M), disable=not verbose):
        wsc_list[m], a_list[m], b_list[m] = wsc_v(X, event_indicators, predict_probs, num_bins, delta, V[m])

    idx_star = np.argmin(np.array(wsc_list))
    a_star = a_list[idx_star]
    b_star = b_list[idx_star]
    v_star = V[idx_star]
    wsc_star = wsc_list[idx_star]
    return wsc_star, v_star, a_star, b_star


def wsc_v(X, event_indicators, predict_probs, num_bins, delta, v):
    """find the worst slab calibration score for a given direction v."""
    optimal = (np.arange(num_bins) + 1) / num_bins
    # this is because there is only (num_bins - 1) degrees of freedom for num_bins
    dof = 1 / (num_bins - 1)

    quantile = np.linspace(1, 0, num_bins + 1)
    position = np.digitize(predict_probs, quantile)
    position[position == 0] = 1  # class probability==1 to the first bin

    binning = np.zeros((len(predict_probs), num_bins))
    for i in range(len(predict_probs)):
        if event_indicators[i]:
            binning[i, position[i] - 1] += 1
        else:
            binning[i, :] = create_censor_binning(predict_probs[i], num_bins)

    n = len(predict_probs)
    z = np.dot(X, v)
    # Compute mass
    z_order = np.argsort(z)
    z_sorted = z[z_order]
    binning_ordered = binning[z_order, :]

    ai_max = int(np.round((1.0 - delta) * n))
    ai_best = 0
    bi_best = n - 1
    xcal_max = xcal_from_hist(np.sum(binning_ordered, axis=0))
    for ai in np.arange(0, ai_max):
        bi_min = np.minimum(ai + int(np.round(delta * n)), n)
        bin = np.cumsum(binning_ordered[ai:n, :], axis=0)
        # normalize the bin
        bin = bin / np.arange(1, n - ai + 1)[:, np.newaxis]
        cum_bin = np.cumsum(bin, axis=1)
        xcals = np.zeros(cum_bin.shape[0])
        # before (bi_min - ai), we don't have enough samples, so we don't need to compute the calibration score
        for i in range(bi_min - ai, cum_bin.shape[0]):
            xcals[i] = dof * np.sum(np.square(cum_bin[i, :] - optimal))

        bi_star = ai + np.argmax(xcals)
        xcal_star = xcals[bi_star - ai]
        if xcal_star > xcal_max:
            ai_best = ai
            bi_best = bi_star
            xcal_max = xcal_star
    return xcal_max, z_sorted[ai_best], z_sorted[bi_best]


def sample_sphere(n, p):
    """Sample n times uniformly from the surface of the p-dimensional unit sphere."""
    # Set random seed for reproducibility if needed (don't need it for now)
    # np.random.seed(random_state)
    v = np.random.randn(p, n)
    v /= np.linalg.norm(v, axis=0)
    return v.T
