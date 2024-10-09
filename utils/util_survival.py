from __future__ import division

import torch
import math
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, PchipInterpolator
from sklearn.utils import shuffle
from typing import Optional, Union
from scipy import interpolate

from skmultilearn.model_selection import iterative_train_test_split

from SurvivalEVAL.Evaluations.util import check_monotonicity, KaplanMeierArea, km_mean
from SurvivalEVAL.Evaluations.custom_types import NumericArrayLike


def format_pred_sksurv(pred_surv):
    time_coordinates = pred_surv[0].x
    surv_prob = np.zeros((len(pred_surv), len(time_coordinates)))
    for i in range(len(pred_surv)):
        if False in (time_coordinates == pred_surv[i].x):
            raise ValueError("Time coordinates are not equal across samples.")
        surv_prob[i, :] = pred_surv[i].y

    # add 0 to time_coordinates and 1 to surv_prob if not present
    if time_coordinates[0] != 0:
        time_coordinates = np.concatenate([np.array([0]), time_coordinates], 0)
        surv_prob = np.concatenate([np.ones([len(surv_prob), 1]), surv_prob], 1)

    return surv_prob, time_coordinates


def make_mono_quantiles(
        quantiles: np.ndarray,
        quan_preds: np.ndarray,
        method: Optional[str] = "ceil",
        seed: Optional[int] = None,
        num_bs: Optional[int] = None
) -> (np.ndarray, np.ndarray):
    """
    Make quantile predictions monotonic and non-negative.
    :param quantiles: np.ndarray of shape (num_quantiles, )
        quantiles to be evaluated
    :param quan_preds: np.ndarray of shape (num_samples, num_quantiles)
        quantile predictions
    :param method: str, optional
        method to make quantile predictions monotonic
    :param seed: int, optional
        random seed
    :param num_bs: int, optional
        number of bootstrap samples to use
    :return:
        quantiles: np.ndarray of shape (num_quantiles, )
            quantiles to be evaluated
        quan_preds: np.ndarray of shape (num_samples, num_quantiles)
            quantile predictions
    """
    # check if quantiles are monotonically increasing
    if np.any(np.sort(quantiles) != quantiles):
        raise ValueError("Defined quantiles must be monotonically increasing.")

    if num_bs is None:
        num_bs = 1000000

    if seed is not None:
        np.random.seed(seed)

    # make sure predictions are non-negative
    quan_preds = np.clip(quan_preds, a_min=0, a_max=None)

    if 0 not in quantiles:
        quantiles = np.insert(quantiles, 0, 0, axis=0)
        quan_preds = np.insert(quan_preds, 0, 0, axis=1)

    if method == "ceil":
        quan_preds = np.maximum.accumulate(quan_preds, axis=1)
    elif method == "floor":
        quan_preds = np.minimum.accumulate(quan_preds[:, ::-1], axis=1)[:, ::-1]
    elif method == "bootstrap":
        # method 1: take too much memory, might cause memory explosion for large dataset
        # need_rearrange = np.any((np.sort(quan_preds, axis=1) != quan_preds), axis=1)
        #
        # extention_at_1 = quan_preds[need_rearrange, -1] / quantiles[-1]
        # inter_lin = interpolate.interp1d(np.r_[quantiles, 1], np.c_[quan_preds[need_rearrange, :], extention_at_1],
        #                                  kind='linear')
        # bootstrap_qf = inter_lin(np.random.uniform(0, 1, num_bs))
        # quan_preds[need_rearrange, :] = np.percentile(bootstrap_qf, 100 * quantiles, axis=1).T
        #
        # method 2: take too much time
        need_rearrange = np.where(np.any((np.sort(quan_preds, axis=1) != quan_preds), axis=1))[0]
        extention_at_1 = quan_preds[:, -1] / quantiles[-1]
        boostrap_samples = np.random.uniform(0, 1, num_bs)
        for idx in need_rearrange:
            inter_lin = interpolate.interp1d(np.r_[quantiles, 1], np.r_[quan_preds[idx, :], extention_at_1[idx]],
                                             kind='linear')
            bootstrap_qf = inter_lin(boostrap_samples)
            quan_preds[idx, :] = np.percentile(bootstrap_qf, 100 * quantiles)
        #
        # method 3: balance between time and memory, but you have to find the right batch size
        # need_rearrange = np.where(np.any((np.sort(quan_preds, axis=1) != quan_preds), axis=1))[0]
        # batch_size = 1024
        # num_batch = need_rearrange.shape[0] // batch_size + (need_rearrange.shape[0] % batch_size > 0)
        # extention_at_1 = quan_preds[:, -1] / quantiles[-1]
        # boostrap_samples = np.random.uniform(0, 1, num_bs)
        # for i in range(num_batch):
        #     idx = need_rearrange[i * batch_size: (i + 1) * batch_size]
        #     inter_lin = interpolate.interp1d(np.r_[quantiles, 1], np.c_[quan_preds[idx, :], extention_at_1[idx]],
        #                                      kind='linear')
        #     bootstrap_qf = inter_lin(boostrap_samples)
        #     quan_preds[idx, :] = np.percentile(bootstrap_qf, 100 * quantiles, axis=1).T
    else:
        raise ValueError(f"Unknown method {method}.")

    # fix some numerical issues
    # In some cases, the quantile predictions can have same value for different quantiles, which will cause the
    # corresponding survival curve to be problematic (multiple percentiles map to a same time).
    # To avoid this, we add a small value to each quantile
    small_values = np.arange(0, quantiles.size) * 1e-10
    quan_preds = quan_preds + small_values

    return quantiles, quan_preds


def compute_decensor_times(test_set, train_set, method="margin", n_sample=1000):
    t_train, e_train = train_set["time"].values, train_set["event"].values.astype(bool)
    n_train = len(t_train)
    km_train = KaplanMeierArea(t_train, e_train)

    t_test, e_test = test_set["time"].values, test_set["event"].values.astype(bool)
    n_test = len(t_test)

    if method == "uncensored":
        # drop censored samples directly
        decensor_set = test_set.drop(test_set[~e_test].index)
        decensor_set.reset_index(drop=True, inplace=True)
        feature_df = decensor_set.drop(["time", "event"], axis=1)
        t = decensor_set["time"].values
        e = decensor_set["event"].values
    elif method == "margin":
        feature_df = test_set.drop(["time", "event"], axis=1)
        best_guesses = t_test.copy().astype(float)
        km_linear_zero = -1 / ((1 - min(km_train.survival_probabilities)) / (0 - max(km_train.survival_times)))
        if np.isinf(km_linear_zero):
            km_linear_zero = max(km_train.survival_times)

        censor_test = t_test[~e_test]
        conditional_mean_t = km_train.best_guess(censor_test)
        conditional_mean_t[censor_test > km_linear_zero] = censor_test[censor_test > km_linear_zero]

        best_guesses[~e_test] = conditional_mean_t
        t = best_guesses
        e = np.ones_like(best_guesses)
    elif method == "PO":
        feature_df = test_set.drop(["time", "event"], axis=1)
        best_guesses = t_test.copy().astype(float)
        events, population_counts = km_train.events.copy(), km_train.population_count.copy()
        times = km_train.survival_times.copy()
        probs = km_train.survival_probabilities.copy()
        # get the discrete time points where the event happens, then calculate the area under those discrete time only
        # this doesn't make any difference for step function, but it does for trapezoid rule.
        unique_idx = np.where(events != 0)[0]
        if unique_idx[-1] != len(events) - 1:
            unique_idx = np.append(unique_idx, len(events) - 1)
        times = times[unique_idx]
        population_counts = population_counts[unique_idx]
        events = events[unique_idx]
        probs = probs[unique_idx]
        sub_expect_time = km_mean(times.copy(), probs.copy())

        # use the idea of dynamic programming to calculate the multiplier of the KM estimator in advances.
        # if we add a new time point to the KM curve, the multiplier before the new time point will be
        # 1 - event_counts / (population_counts + 1), and the multiplier after the new time point will be
        # the same as before.
        multiplier = 1 - events / population_counts
        multiplier_total = 1 - events / (population_counts + 1)

        for i in range(n_test):
            if e_test[i] != 1:
                total_multiplier = multiplier.copy()
                insert_index = np.searchsorted(times, t_test[i], side='right')
                total_multiplier[:insert_index] = multiplier_total[:insert_index]
                survival_probabilities = np.cumprod(total_multiplier)
                if insert_index == len(times):
                    times_addition = np.append(times, t_test[i])
                    survival_probabilities_addition = np.append(survival_probabilities, survival_probabilities[-1])
                    total_expect_time = km_mean(times_addition, survival_probabilities_addition)
                else:
                    total_expect_time = km_mean(times, survival_probabilities)
                best_guesses[i] = (n_train + 1) * total_expect_time - n_train * sub_expect_time

        t = best_guesses
        e = np.ones_like(best_guesses)
    elif method == "sampling":
        # repeat each sample n_sample times,
        # for event subject, the event time will be the same for n_sample times;
        # for censored subject, the "fake" event time will be sampled from the conditional KM curve,
        # the conditional KM curve is the KM distribution (km_train) given we know the subject is censored at time c
        # and make the censor bit to 1
        feature_df = test_set.drop(["time", "event"], axis=1)
        t = np.repeat(test_set["time"].values, n_sample)
        uniq_times = km_train.survival_times
        surv = km_train.survival_probabilities
        last_time = km_train.km_linear_zero
        if uniq_times[0] != 0:
            uniq_times = np.insert(uniq_times, 0, 0, axis=0)
            surv = np.insert(surv, 0, 1, axis=0)

        for i in range(n_test):
            # x_i = x_test[i, :]
            if e_test[i] != 1:
                s_prob = km_train.predict(t_test[i])
                if s_prob <= 0:
                    # if the survival probability is 0, then the conditional KM curve is 0,
                    # we can't sample from it, so we just use the censor time
                    t[i * n_sample:(i + 1) * n_sample] = t_test[i]
                else:
                    cond_surv = surv / s_prob
                    cond_surv = np.clip(cond_surv, 0, 1)
                    cond_cdf = 1 - cond_surv
                    cond_pdf = np.diff(np.append(cond_cdf, 1))

                    # sample from the conditional KM curve
                    surrogate_t = np.random.choice(uniq_times, size=n_sample, p=cond_pdf)

                    if last_time != uniq_times[-1]:
                        need_extension = surrogate_t == uniq_times[-1]
                        surrogate_t[need_extension] = np.random.uniform(uniq_times[-1], last_time, need_extension.sum())

                    t[i * n_sample:(i + 1) * n_sample] = surrogate_t

        e = np.ones_like(t)

    else:
        raise ValueError(f"Unknown method {method}.")

    return feature_df, t, e


def compute_unique_counts(
        event: torch.Tensor,
        time: torch.Tensor,
        order: Optional[torch.Tensor] = None):
    """Count right censored and uncensored samples at each unique time point.

    Parameters
    ----------
    event : array
        Boolean event indicator.

    time : array
        Survival time or time of censoring.

    order : array or None
        Indices to order time in ascending order.
        If None, order will be computed.

    Returns
    -------
    times : array
        Unique time points.

    n_events : array
        Number of events at each time point.

    n_at_risk : array
        Number of samples that have not been censored or have not had an event at each time point.

    n_censored : array
        Number of censored samples at each time point.
    """
    n_samples = event.shape[0]

    if order is None:
        order = torch.argsort(time)

    uniq_times = torch.empty(n_samples, dtype=time.dtype, device=time.device)
    uniq_events = torch.empty(n_samples, dtype=torch.int, device=time.device)
    uniq_counts = torch.empty(n_samples, dtype=torch.int, device=time.device)

    i = 0
    prev_val = time[order[0]]
    j = 0
    while True:
        count_event = 0
        count = 0
        while i < n_samples and prev_val == time[order[i]]:
            if event[order[i]]:
                count_event += 1

            count += 1
            i += 1

        uniq_times[j] = prev_val
        uniq_events[j] = count_event
        uniq_counts[j] = count
        j += 1

        if i == n_samples:
            break

        prev_val = time[order[i]]

    uniq_times = uniq_times[:j]
    uniq_events = uniq_events[:j]
    uniq_counts = uniq_counts[:j]
    n_censored = uniq_counts - uniq_events

    # offset cumulative sum by one
    total_count = torch.cat([torch.tensor([0], device=uniq_counts.device), uniq_counts], dim=0)
    n_at_risk = n_samples - torch.cumsum(total_count, dim=0)

    return uniq_times, uniq_events, n_at_risk[:-1], n_censored


def baseline_hazard(
        logits: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Calculate the baseline cumulative hazard function and baseline survival function using Breslow estimator
    :param logits: logit outputs calculated from the Cox-based network using training data.
    :param time: Survival time of training data.
    :param event: Survival indicator of training data.
    :return:
    uniq_times: time bins correspond of the baseline hazard/survival.
    cum_baseline_hazard: cumulative baseline hazard
    baseline_survival: baseline survival curve.
    """
    risk_score = torch.exp(logits)
    order = torch.argsort(time)
    risk_score = risk_score[order]
    uniq_times, n_events, n_at_risk, _ = compute_unique_counts(event, time, order)

    divisor = torch.empty(n_at_risk.shape, dtype=torch.float, device=n_at_risk.device)
    value = torch.sum(risk_score)
    divisor[0] = value
    k = 0
    for i in range(1, len(n_at_risk)):
        d = n_at_risk[i - 1] - n_at_risk[i]
        value -= risk_score[k:(k + d)].sum()
        k += d
        divisor[i] = value

    assert k == n_at_risk[0] - n_at_risk[-1]

    hazard = n_events / divisor
    # Make sure the survival curve always starts at 1
    if 0 not in uniq_times:
        uniq_times = torch.cat([torch.tensor([0]).to(uniq_times.device), uniq_times], 0)
        hazard = torch.cat([torch.tensor([0]).to(hazard.device), hazard], 0)
    # TODO: torch.cumsum with cuda array will generate a non-monotonic array. Need to update when torch fix this bug
    # See issue: https://github.com/pytorch/pytorch/issues/21780
    cum_baseline_hazard = torch.cumsum(hazard.cpu(), dim=0).to(hazard.device)
    baseline_survival = torch.exp(- cum_baseline_hazard)
    if baseline_survival.isinf().any() or (not check_monotonicity(baseline_survival)):
        print(f"Baseline survival contains \'inf\', need attention. \n"
              f"Baseline survival distribution: {baseline_survival}")
        last_zero = torch.where(baseline_survival == 0)[0][-1].item()
        baseline_survival[last_zero + 1:] = 0
    return uniq_times, hazard, cum_baseline_hazard, baseline_survival


def reformat_survival(
        dataset: pd.DataFrame,
        time_bins: NumericArrayLike
) -> (torch.Tensor, torch.Tensor):
    x = torch.tensor(dataset.drop(["time", "event"], axis=1).values, dtype=torch.float)
    y = encode_survival(dataset["time"].values, dataset["event"].values, time_bins)
    return x, y


def extract_survival(
        df: pd.DataFrame,
        discrete_bins: Optional[NumericArrayLike] = None
) -> (torch.Tensor, torch.Tensor, np.ndarray, np.ndarray):
    x = torch.from_numpy(df.drop(columns=['time', 'event']).values)
    t, e = torch.from_numpy(df['time'].values), torch.from_numpy(df['event'].values)
    if discrete_bins is not None:
        # discrete time models
        y = encode_survival(t, e, discrete_bins)
    else:
        # continuous time models
        y = torch.stack([t, e], dim=1)
    return x, y, t, e


def encode_survival(
        time: Union[float, int, NumericArrayLike],
        event: Union[int, bool, NumericArrayLike],
        bins: NumericArrayLike
) -> torch.Tensor:
    """Encodes survival time and event indicator in the format
    required for MTLR training.

    For uncensored instances, one-hot encoding of binned survival time
    is generated. Censoring is handled differently, with all possible
    values for event time encoded as 1s. For example, if 5 time bins are used,
    an instance experiencing event in bin 3 is encoded as [0, 0, 0, 1, 0], and
    instance censored in bin 2 as [0, 0, 1, 1, 1]. Note that an additional
    'catch-all' bin is added, spanning the range `(bins.max(), inf)`.

    Parameters
    ----------
    time : Union[float, int, np.ndarray, torch.Tensor]
        Survival times.
    event : Union[int, bool, np.ndarray, torch.Tensor]
        Event indicators.
    bins : np.ndarray
        Time bins.
    Returns
    -------
    torch.Tensor
        Encoded survival times.
    """
    if isinstance(time, (float, int, np.ndarray)):
        time = np.atleast_1d(time)
        time = torch.tensor(time)
    if isinstance(event, (int, bool, np.ndarray)):
        event = np.atleast_1d(event)
        event = torch.tensor(event)

    if isinstance(bins, np.ndarray):
        bins = torch.tensor(bins)

    try:
        device = bins.device
    except AttributeError:
        device = "cpu"

    time = np.clip(time, 0, bins.max())
    # add extra bin [max_time, inf) at the end
    y = torch.zeros((time.shape[0], bins.shape[0] + 1),
                    dtype=torch.float,
                    device=device)
    # For some reason, the `right` arg in torch.bucketize
    # works in the _opposite_ way as it does in numpy,
    # so we need to set it to True
    bin_idxs = torch.bucketize(time, bins, right=True)
    for i, (bin_idx, e) in enumerate(zip(bin_idxs, event)):
        if e == 1:
            y[i, bin_idx] = 1
        else:
            y[i, bin_idx:] = 1
    return y.squeeze()


def make_time_bins(
        times: NumericArrayLike,
        num_bins: Optional[int] = None,
        use_quantiles: bool = True,
        event: Optional[NumericArrayLike] = None,
        add_last_time: Optional[bool] = False
) -> torch.Tensor:
    """Creates the bins for survival time discretisation.

    By default, sqrt(num_observation) bins corresponding to the quantiles of
    the survival time distribution are used, as in https://github.com/haiderstats/MTLR.

    Parameters
    ----------
    times
        Array or tensor of survival times.
    num_bins
        The number of bins to use. If None (default), sqrt(num_observations)
        bins will be used.
    use_quantiles
        If True, the bin edges will correspond to quantiles of `times`
        (default). Otherwise, generates equally-spaced bins.
    event
        Array or tensor of event indicators. If specified, only samples where
        event == 1 will be used to determine the time bins.
    add_last_time
        If True, the last time bin will be added to the end of the time bins.
    Returns
    -------
    torch.Tensor
        Tensor of bin edges.
    """
    if event is not None:
        times = times[event == 1]
    if num_bins is None:
        num_bins = math.ceil(math.sqrt(len(times)))
    if use_quantiles:
        # NOTE we should switch to using torch.quantile once it becomes
        # available in the next version
        bins = np.unique(np.quantile(times, np.linspace(0, 1, num_bins)))
    else:
        bins = np.linspace(times.min(), times.max(), num_bins)
    bins = torch.tensor(bins, dtype=torch.float)
    if add_last_time:
        bins = torch.cat([bins, torch.tensor([times.max()])])
    return bins


def survival_stratified_cv(
        dataset: pd.DataFrame,
        event_times: np.ndarray,
        event_indicators: np.ndarray,
        number_folds: int = 5
) -> list:
    event_times, event_indicators = event_times.tolist(), event_indicators.tolist()
    assert len(event_indicators) == len(event_times)

    indicators_and_times = list(zip(event_indicators, event_times))
    sorted_idx = [i[0] for i in sorted(enumerate(indicators_and_times), key=lambda v: (v[1][0], v[1][1]))]

    folds = [[sorted_idx[0]], [sorted_idx[1]], [sorted_idx[2]], [sorted_idx[3]], [sorted_idx[4]]]
    for i in range(5, len(sorted_idx)):
        fold_number = i % number_folds
        folds[fold_number].append(sorted_idx[i])

    training_sets = [dataset.drop(folds[i], axis='index').reset_index(drop=True) for i in range(number_folds)]
    testing_sets = [dataset.iloc[folds[i], :].reset_index(drop=True) for i in range(number_folds)]

    cross_validation_set = list(zip(training_sets, testing_sets))
    return cross_validation_set


def multilabel_train_test_split(x, y, test_size, random_state=None):
    """Iteratively stratified train/test split
    (Add random_state to scikit-multilearn iterative_train_test_split function)
    See this paper for details: https://link.springer.com/chapter/10.1007/978-3-642-23808-6_10
    # TODO: the current function `iterative_train_test_split` is not efficient, need to find a better way to do this.
    See https://github.com/scikit-multilearn/scikit-multilearn/issues/202
    """
    x, y = shuffle(x, y, random_state=random_state)
    x_train, y_train, x_test, y_test = iterative_train_test_split(x, y, test_size=test_size)
    return x_train, y_train, x_test, y_test


def survival_data_split(
        df: pd.DataFrame,
        stratify_colname: str = 'event',
        frac_train: float = 0.5,
        frac_val: float = 0.0,
        frac_test: float = 0.5,
        random_state: int = None
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    assert frac_train >= 0 and frac_val >= 0 and frac_test >= 0, "Check train validation test fraction."
    frac_sum = frac_train + frac_val + frac_test
    frac_train = frac_train / frac_sum
    frac_val = frac_val / frac_sum
    frac_test = frac_test / frac_sum

    x = df.values  # Contains all columns.
    columns = df.columns
    if stratify_colname == 'event':
        stra_lab = df[stratify_colname]
    elif stratify_colname == 'time':
        stra_lab = df[stratify_colname]
        bins = np.linspace(start=stra_lab.min(), stop=stra_lab.max(), num=20)
        stra_lab = np.digitize(stra_lab, bins, right=True)
    elif stratify_colname == "both":
        t = df["time"]
        bins = np.linspace(start=t.min(), stop=t.max(), num=20)
        t = np.digitize(t, bins, right=True)
        e = df["event"]
        stra_lab = np.stack([t, e], axis=1)
    else:
        raise ValueError("unrecognized stratify policy")

    x_train_val, y_train_val, x_test, _ = multilabel_train_test_split(x, y=stra_lab, test_size=frac_test,
                                                                      random_state=random_state)
    if frac_val == 0:
        x_train, x_val = x_train_val, []
    else:
        x_train, _, x_val, _ = multilabel_train_test_split(x_train_val, y=y_train_val,
                                                           test_size=frac_val / (frac_val + frac_train),
                                                           random_state=random_state)
    df_train = pd.DataFrame(data=x_train, columns=columns)
    df_val = pd.DataFrame(data=x_val, columns=columns)
    df_test = pd.DataFrame(data=x_test, columns=columns)
    assert len(df) == len(df_train) + len(df_val) + len(df_test)
    return df_train, df_val, df_test


def xcal_from_hist(d_cal_hist: np.ndarray):
    """
    Compute the x-calibration score from the D-calibration histogram.
    """
    # get bin number
    n_bins = d_cal_hist.shape[0]
    # normalize the histogram
    d_cal_hist = d_cal_hist / d_cal_hist.sum()
    # compute the x-calibration score
    optimal = np.ones_like(d_cal_hist) / n_bins
    # 1/(n_bins-1) is because there is only (n_bins-1) degrees of freedom for n_bins
    x_cal = (1 / (n_bins - 1)) * np.sum(np.square(d_cal_hist.cumsum() - optimal.cumsum()))
    return x_cal


def survival_to_quantile(surv_prob, time_coordinates, quantile_levels, interpolate='Pchip'):
    if interpolate == 'Linear':
        Interpolator = interp1d
    elif interpolate == 'Pchip':
        Interpolator = PchipInterpolator
    else:
        raise ValueError(f"Unknown interpolation method: {interpolate}")

    cdf = 1 - surv_prob
    slope = cdf[:, -1] / time_coordinates[:, -1]
    assert cdf.shape == time_coordinates.shape, "CDF and time coordinates have different shapes."
    quantile_predictions = np.empty((cdf.shape[0], quantile_levels.shape[0]))
    for i in range(cdf.shape[0]):
        # fit a scipy interpolation function to the cdf
        cdf_i = cdf[i, :]
        time_coordinates_i = time_coordinates[i, :]
        # remove duplicates in cdf_i (x-axis), otherwise Interpolator will raise an error
        # here we only keep the first occurrence of each unique value
        cdf_i, idx = np.unique(cdf_i, return_index=True)
        time_coordinates_i = time_coordinates_i[idx]
        interp = Interpolator(cdf_i, time_coordinates_i)

        # if the quantile level is beyond last cdf, we extrapolate the
        beyond_last_idx = np.where(quantile_levels > cdf_i[-1])[0]
        quantile_predictions[i] = interp(quantile_levels)
        quantile_predictions[i, beyond_last_idx] = quantile_levels[beyond_last_idx] / slope[i]

    # sanity checks
    assert np.all(quantile_predictions >= 0), "Quantile predictions contain negative."
    assert check_monotonicity(quantile_predictions), "Quantile predictions are not monotonic."
    return quantile_predictions
