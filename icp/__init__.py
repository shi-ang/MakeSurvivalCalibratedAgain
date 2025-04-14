from __future__ import division

from collections import defaultdict
from functools import partial
from abc import abstractmethod
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from utils.util_survival import compute_decensor_times
from icp.scorer import QuantileRegressionNC, SurvivalPredictionNC
from icp.error_functions import OnsSideQuantileRegErrFunc


class ConformalSurvivalBase(BaseEstimator):
    """Abstract class for survival conformal prediction."""
    def __init__(self, nc_function, decensor_method: str, n_quantiles: int, n_sample: int):
        self.nc_function = nc_function
        self.train_data = None
        self.cal_data = None
        self.feature_names = None
        self.decensor_method = decensor_method
        self.cal_scores = None

        assert isinstance(n_quantiles, int) and n_quantiles > 0, "n_quantiles must be a positive integer"
        self.n_quantiles = n_quantiles
        self.quantile_levels = np.linspace(1 / (self.n_quantiles + 1), self.n_quantiles / (self.n_quantiles + 1),
                                           self.n_quantiles)

        self.n_sample = n_sample

    def fit(self, data_train, data_val):
        self.train_data = data_train
        self.feature_names = data_train.drop(['time', 'event'], axis=1).columns.tolist()
        self.nc_function.fit(self.train_data, data_val)

    @abstractmethod
    def calibrate(self, data_val, increment=False):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    def _update_calibration_set(self, data: pd.DataFrame, increment):
        if increment and self.cal_data is not None:
            self.cal_data = pd.concat([self.cal_data, data])
        else:
            self.cal_data = data


class ConformalSurvDist(ConformalSurvivalBase):
    """Conformalized survival distribution.

    Shi-ang Qi, Yakun Yu, Russell Greiner. Conformalized Survival Distributions: A Generic Post-Process to Increase
    Calibration. ICML 2024.
    https://proceedings.mlr.press/v235/qi24a.html
    """
    def __init__(
            self,
            nc_function: QuantileRegressionNC,
            condition=None,
            decensor_method: str = 'margin',
            n_quantiles: int = None,
            n_sample: int = 1000
    ):
        # Number of quantile levels. Should be an integer.
        # If ``None``, then we use the 9 default quantile levels (0.1, 0.2, ..., 0.9).
        if n_quantiles is None:
            self.n_quantiles = 9

        # Check if condition-parameter is the default function (i.e.,
        # lambda x: 0). This is so we can safely clone the object without
        # the clone accidentally having self.conditional = True.
        default_condition = lambda x: 0
        is_default = callable(condition) and (condition.__code__.co_code == default_condition.__code__.co_code)

        if is_default:
            self.condition = condition
            self.conditional = False
        elif callable(condition):
            self.condition = condition
            self.conditional = True
        else:
            self.condition = lambda x: 0
            self.conditional = False

        self.categories = None

        super().__init__(nc_function, decensor_method, n_quantiles, n_sample)

    def calibrate(self, data_val, increment=False):
        self._update_calibration_set(data_val, increment)

        features, t, e = compute_decensor_times(self.cal_data, self.train_data,
                                                method=self.decensor_method, n_sample=self.n_sample)

        if self.conditional:
            # Not tested yet for this current project
            category_map = self.condition(features)
            self.categories = np.unique(category_map)
            self.cal_scores = defaultdict(partial(np.ndarray, 0))

            for cond in self.categories:
                idx = (category_map == cond)
                cal_scores = self.nc_function.score(feature_df=features[idx, :], t=t[idx], e=e[idx],
                                                    quantile_levels=self.quantile_levels,
                                                    n_sample=self.n_sample if self.decensor_method == 'sampling' else None)
                self.cal_scores[cond] = np.sort(cal_scores, 0)[::-1]
        else:
            self.categories = np.array([0])
            cal_scores = self.nc_function.score(feature_df=features, t=t, e=e,
                                                quantile_levels=self.quantile_levels,
                                                n_sample=self.n_sample if self.decensor_method == 'sampling' else None)
            self.cal_scores = {0: np.sort(cal_scores, 0)[::-1]}

    def predict(self, x):
        """Predict the output values for a set of input patterns.

        Parameters
        ----------
        x : numpy array of shape [n_samples, n_features]
            Inputs of patters for which to predict output values.

        Returns
        -------
        p : numpy array of shape [n_samples, n_quantiles]
            If `quantile_levels` is ``None``, then p contains the interval (minimum
            and maximum boundaries) for each test pattern, and each quantile
            level (0.1, 0.2, ..., 0.9).
        """
        quan_pred = np.zeros((x.shape[0], self.n_quantiles + 1))

        condition_map = np.array([self.condition((x[i, :], None)) for i in range(x.shape[0])])

        for condition in self.categories:
            idx = condition_map == condition
            if np.sum(idx) > 0:
                p = self.nc_function.predict(x[idx, :], self.cal_scores[condition],
                                             self.feature_names, self.quantile_levels)
                quan_pred[idx, :] = p

        if 0 not in self.quantile_levels:
            quan_levels = np.insert(self.quantile_levels, 0, 0)
        else:
            quan_levels = self.quantile_levels

        return quan_levels, quan_pred


class CSDiPOT(ConformalSurvivalBase):
    """Conformalized survival distribution using individual probability at observed time.

    Shi-ang Qi, Yakun Yu, Russell Greiner. Toward Conditional Distribution Calibration in Survival Prediction.
    NeurIPS 2024.
    """
    def __init__(
            self,
            nc_function: SurvivalPredictionNC,
            decensor_method: str = 'sampling',
            n_percentile: int = None,
            n_sample: int = 1000
    ):
        if n_percentile is None:
            n_percentile = 9

        super().__init__(nc_function, decensor_method, n_percentile, n_sample)

    def calibrate(self, data_val, increment=False):
        self._update_calibration_set(data_val, increment)

        if self.decensor_method == 'sampling':
            features, t, e = data_val.drop(['time', 'event'], axis=1), data_val['time'], data_val['event']
            cal_scores = self.nc_function.score(feature_df=features, t=t, e=e)
        else:
            features, t, e = compute_decensor_times(self.cal_data, self.train_data, method=self.decensor_method)
            cal_scores = self.nc_function.score(feature_df=features, t=t, e=e)
        self.cal_scores = - np.sort(- cal_scores)

    def predict(self, x):
        quan_pred = self.nc_function.predict(x, self.cal_scores, self.feature_names, self.quantile_levels)

        # check if quan_pred is strictly monotonic increasing
        if not np.all(np.diff(quan_pred, axis=1) > 0):
            small_values = np.arange(0, quan_pred.shape[1]) * 1e-10
            quan_pred += small_values

        if 0 not in self.quantile_levels:
            quan_levels = np.insert(self.quantile_levels, 0, 0)
            quan_pred = np.insert(quan_pred, 0, 0, axis=1)
        else:
            quan_levels = self.quantile_levels

        return quan_levels, quan_pred
