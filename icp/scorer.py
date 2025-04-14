from __future__ import division
import argparse
import abc
import numpy as np
import sklearn.base
import torch
import torchtuples as tt
import pandas as pd
import os

from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis
from lifelines.fitters.weibull_aft_fitter import WeibullAFTFitter
from model import CoxPH, MTLR, CQRNN, LogNormalNN
from pycox.models import DeepHitSingle, CoxTime

from icp.error_functions import AbsErrorErrFunc, RegressionErrFunc, OnsSideQuantileRegErrFunc, ProbAtEventTimeErrFunc
from utils import pad_tensor
from utils.util_survival import make_mono_quantiles, format_pred_sksurv, survival_to_quantile
from SurvivalEVAL.Evaluations.util import check_monotonicity


class BaseScorer(sklearn.base.BaseEstimator):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(BaseScorer, self).__init__()

    @abc.abstractmethod
    def fit(self, x, y):
        pass

    @abc.abstractmethod
    def score(self, x, y=None):
        pass


class BaseModelNc(BaseScorer):
    """Base class for nonconformity scorers based on an underlying model.

    Parameters
    ----------
    model : ClassifierAdapter or RegressorAdapter
        Underlying classification model used for calculating nonconformity
        scores.

    err_func : ClassificationErrFunc or RegressionErrFunc
        Error function object.

    normalizer : BaseScorer
        Normalization model.

    beta : float
        Normalization smoothing parameter. As the beta-value increases,
        the normalized nonconformity function approaches a non-normalized
        equivalent.
    """

    def __init__(self, model, err_func, normalizer=None, beta=1e-6):
        super(BaseModelNc, self).__init__()
        self.err_func = err_func
        self.model = model
        self.normalizer = normalizer
        self.beta = beta

        # If we use sklearn.base.clone (e.g., during cross-validation),
        # object references get jumbled, so we need to make sure that the
        # normalizer has a reference to the proper model adapter, if applicable.
        if (self.normalizer is not None and
                hasattr(self.normalizer, 'base_model')):
            self.normalizer.base_model = self.model

        self.last_x, self.last_y = None, None
        self.last_prediction = None
        self.clean = False

    def fit(self, x, y):
        """Fits the underlying model of the nonconformity scorer.

        Parameters
        ----------
        x : numpy array of shape [n_samples, n_features]
            Inputs of examples for fitting the underlying model.

        y : numpy array of shape [n_samples]
            Outputs of examples for fitting the underlying model.

        Returns
        -------
        None
        """
        self.model.fit(x, y)
        if self.normalizer is not None:
            self.normalizer.fit(x, y)
        self.clean = False

    def score(self, x, y=None):
        """Calculates the nonconformity score of a set of samples.

        Parameters
        ----------
        x : numpy array of shape [n_samples, n_features]
            Inputs of examples for which to calculate a nonconformity score.

        y : numpy array of shape [n_samples]
            Outputs of examples for which to calculate a nonconformity score.

        Returns
        -------
        nc : numpy array of shape [n_samples]
            Nonconformity scores of samples.
        """
        prediction = self.model.predict(x)
        n_test = x.shape[0]
        if self.normalizer is not None:
            norm = self.normalizer.score(x) + self.beta
        else:
            norm = np.ones(n_test)
        if prediction.ndim > 1:
            ret_val = self.err_func.apply(prediction, y)
        else:
            ret_val = self.err_func.apply(prediction, y) / norm
        return ret_val


class RegressorNc(BaseModelNc):
    """Nonconformity scorer using an underlying regression model.

    Parameters
    ----------
    model : RegressorAdapter
        Underlying regression model used for calculating nonconformity scores.

    err_func : RegressionErrFunc
        Error function object.

    normalizer : BaseScorer
        Normalization model.

    beta : float
        Normalization smoothing parameter. As the beta-value increases,
        the normalized nonconformity function approaches a non-normalized
        equivalent.

    Attributes
    ----------
    model : RegressorAdapter
        Underlying model object.

    err_func : RegressionErrFunc
        Scorer function used to calculate nonconformity scores.

    See also
    --------
    ProbEstClassifierNc, NormalizedRegressorNc
    """

    def __init__(self,
                 model,
                 err_func=AbsErrorErrFunc(),
                 normalizer=None,
                 beta=1e-6):
        super(RegressorNc, self).__init__(model,
                                          err_func,
                                          normalizer,
                                          beta)

    def predict(self, x, nc, significance=None):
        """Constructs prediction intervals for a set of test examples.

        Predicts the output of each test pattern using the underlying model,
        and applies the (partial) inverse nonconformity function to each
        prediction, resulting in a prediction interval for each test pattern.

        Parameters
        ----------
        x : numpy array of shape [n_samples, n_features]
            Inputs of patters for which to predict output values.

        nc : numpy array of shape [n_calibration_samples]
            Nonconformity scores obtained for conformal predictor.

        significance : float
            Significance level (maximum allowed error rate) of predictions.
            Should be a float between 0 and 1. If ``None``, then intervals for
            all significance levels (0.01, 0.02, ..., 0.99) are output in a
            3d-matrix.

        Returns
        -------
        p : numpy array of shape [n_samples, 2] or [n_samples, 2, 99]
            If significance is ``None``, then p contains the interval (minimum
            and maximum boundaries) for each test pattern, and each significance
            level (0.01, 0.02, ..., 0.99). If significance is a float between
            0 and 1, then p contains the prediction intervals (minimum and
            maximum	boundaries) for the set of test patterns at the chosen
            significance level.
        """
        n_test = x.shape[0]
        prediction = self.model.predict(x)
        if self.normalizer is not None:
            norm = self.normalizer.score(x) + self.beta
        else:
            norm = np.ones(n_test)

        if significance:
            intervals = np.zeros((x.shape[0], 2))
            err_dist = self.err_func.apply_inverse(nc, significance)
            err_dist = np.hstack([err_dist] * n_test)
            if prediction.ndim > 1:  # CQR
                intervals[:, 0] = prediction[:, 0] - err_dist[0, :]
                intervals[:, 1] = prediction[:, -1] + err_dist[1, :]
            else:  # regular conformal prediction
                err_dist *= norm
                intervals[:, 0] = prediction - err_dist[0, :]
                intervals[:, 1] = prediction + err_dist[1, :]

            return intervals
        else:  # Not tested for CQR
            significance = np.arange(0.01, 1.0, 0.01)
            intervals = np.zeros((x.shape[0], 2, significance.size))

            for i, s in enumerate(significance):
                err_dist = self.err_func.apply_inverse(nc, s)
                err_dist = np.hstack([err_dist] * n_test)
                err_dist *= norm

                intervals[:, 0, i] = prediction - err_dist[0, :]
                intervals[:, 1, i] = prediction + err_dist[0, :]

            return intervals


class QuantileRegressionNC(sklearn.base.BaseEstimator):
    """Nonconformity scorer using an underlying censored quantile regresson model."""

    def __init__(
            self,
            model,
            args=argparse.Namespace
    ):
        super(QuantileRegressionNC, self).__init__()
        self.model = model
        self.err_func = OnsSideQuantileRegErrFunc()
        self.args = args

    def fit(self, train_set: pd.DataFrame, val_set: pd.DataFrame):
        """Fits the underlying model of the nonconformity scorer."""
        if (isinstance(self.model, CoxPH) or isinstance(self.model, MTLR)
                or isinstance(self.model, CQRNN) or isinstance(self.model, LogNormalNN)):
            folder = 'logs/QuantileRegressionNC'
            # create folder if it does not exist
            if not os.path.exists(folder):
                os.makedirs(folder)
            self.model.fit(train_set, val_set, device=self.args.device, batch_size=self.args.batch_size,
                           epochs=self.args.n_epochs, lr=self.args.lr, lr_min=1e-3 * self.args.lr,
                           weight_decay=self.args.weight_decay, early_stop=self.args.early_stop,
                           fname=folder + f'/{self.model.__class__.__name__}', verbose=self.args.verbose)
        elif isinstance(self.model, DeepHitSingle) or isinstance(self.model, CoxTime):
            x_train = train_set.drop(['time', 'event'], axis=1).values
            t_train, e_train = train_set['time'].values, train_set['event'].values
            x_val = val_set.drop(['time', 'event'], axis=1).values
            t_val, e_val = val_set['time'].values, val_set['event'].values
            y_train = self.model.label_transform.transform(t_train, e_train)
            y_val = self.model.label_transform.transform(t_val, e_val)
            val = (x_val, y_val)
            val_size = x_val.shape[0]

            self.model.optimizer.set_lr(self.args.lr)
            self.model.optimizer.set('weight_decay', self.args.weight_decay)
            if self.args.early_stop:
                callbacks = [tt.callbacks.EarlyStopping()]
            else:
                callbacks = None

            self.model.fit(input=x_train, target=y_train, batch_size=self.args.batch_size, epochs=self.args.n_epochs,
                           callbacks=callbacks, verbose=self.args.verbose, val_data=val, val_batch_size=val_size)
            if isinstance(self.model, CoxTime):
                self.model.compute_baseline_hazards()
        elif isinstance(self.model, WeibullAFTFitter):
            if self.args.use_train:
                # if we choose to use the training set for calibration, we need to combine the train and val sets
                train_set = pd.concat([train_set, val_set], ignore_index=True)
            self.model.fit(train_set, duration_col='time', event_col='event')
        elif isinstance(self.model, ComponentwiseGradientBoostingSurvivalAnalysis):
            if self.args.use_train:
                # if we choose to use the training set for calibration, we need to combine the train and val sets
                train_set = pd.concat([train_set, val_set], ignore_index=True)
            x_train = train_set.drop(["time", "event"], axis=1).values
            t_train, e_train = train_set["time"].values, train_set["event"].values
            y_train = np.empty(dtype=[('cens', bool), ('time', np.float64)], shape=t_train.shape[0])
            y_train['cens'] = e_train
            y_train['time'] = t_train
            self.model.fit(x_train, y_train)

    def score(
            self,
            feature_df: pd.DataFrame,
            t: np.ndarray,
            e: np.ndarray,
            quantile_levels: np.ndarray,
            n_sample: int = 1000
    ):
        """Calculates the nonconformity score of a set of samples.

        Parameters
        ----------
        feature_df: pandas DataFrame of shape [n_samples, n_features]
            Inputs of examples for which to calculate a nonconformity score.
        t: numpy array of shape [n_samples]
            Times of examples.
        e: numpy array of shape [n_samples]
            Event indicators of examples.
        quantile_levels: numpy array of shape [n_significance_levels]
            Significance levels (maximum allowed error rate) of predictions.
        method: str
            Decensoring method to use. See `compute_decensor_times` in `utils/util_survival.py` for details.
        Returns
        -------
        conformal_scores : numpy array of shape [n_samples]
            conformity scores of samples.
        """
        x = feature_df.values
        x_names = feature_df.columns.tolist()
        y = np.stack([t, e], axis=1)

        quantile_predictions = self.predict_nc(x, quantile_levels, x_names)

        if n_sample is not None:
            quantile_predictions = np.repeat(quantile_predictions, n_sample, axis=0)

        assert quantile_predictions.shape[0] == y.shape[0], "Sample size does not match."

        conformal_scores = self.err_func.apply(quantile_predictions, y)
        return conformal_scores

    def predict(
            self,
            x: np.ndarray,
            conformal_scores: np.ndarray,
            feature_names: list[str] = None,
            quantile_levels=None
    ):
        quantile_predictions = self.predict_nc(x, quantile_levels, feature_names)

        error_dist = self.err_func.apply_inverse(conformal_scores, quantile_levels)

        quantile_predictions = quantile_predictions - error_dist
        quantile_levels, quantile_predictions = make_mono_quantiles(quantile_levels, quantile_predictions,
                                                                    method=self.args.mono_method, seed=self.args.seed)
        # sanity checks
        assert np.all(quantile_predictions >= 0), "Quantile predictions contain negative."
        assert check_monotonicity(quantile_predictions), "Quantile predictions are not monotonic."

        return quantile_predictions

    def predict_nc(
            self,
            x: np.ndarray,
            quantile_levels: np.ndarray,
            feature_names: list[str] = None
    ) -> np.ndarray:
        """
        Predict the nonconformity survival curves for a given feature matrix x

        :param x: numpy array of shape [n_samples, n_features]
            feature matrix
        :param feature_names: list of strings
            feature names. Only used for lifelines models.
        :param quantile_levels: numpy array of shape [n_quantiles]
            quantile levels to predict
        :return:
        Quantile predictions for the survival curves
            quantile_predictions: numpy array of shape [n_samples, n_quantiles]
        """
        if isinstance(self.model, CQRNN):
            x = torch.from_numpy(x).float().to(self.args.device)
            quantile_predictions = self.model.predict_quantiles(x).cpu().numpy()
            q_levels = self.model.quan_levels.cpu().numpy()
            # It seems redundant to do the monotonicity here, we will do it after conformalization.
            # q_levels, quantile_predictions = make_mono_quantiles(q_levels, quantile_predictions,
            #                                                      method=self.args.mono_method, seed=self.args.seed)
            # q_levels = q_levels[1:]
            assert np.abs(q_levels - quantile_levels).max() < 1e-4, "Quantile levels do not match."
            # quantile_predictions = quantile_predictions[:, 1:]
        else:
            # using batch prediction to avoid memory overflow, choose the largest batch size
            # that does not cause memory overflow, this shouldn't impact the performance
            batch_size = 16384
            num_batches = x.shape[0] // batch_size + (x.shape[0] % batch_size > 0)
            quantile_batchs = []
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, x.shape[0])
                x_batch = x[start_idx:end_idx]

                if isinstance(self.model, CoxPH) or isinstance(self.model, MTLR) or isinstance(self.model, LogNormalNN):
                    x_batch = torch.from_numpy(x_batch).float().to(self.args.device)
                    surv_prob = self.model.predict_survival(x_batch).cpu().numpy()
                    time_coordinates = self.model.time_bins
                    if isinstance(self.model, MTLR):
                        time_coordinates = pad_tensor(time_coordinates, 0, where='start')
                    time_coordinates = time_coordinates.cpu().numpy()
                elif isinstance(self.model, DeepHitSingle) or isinstance(self.model, CoxTime):
                    surv_df = self.model.predict_surv_df(x_batch)
                    time_coordinates = surv_df.index.values
                    surv_prob = surv_df.values.T
                elif isinstance(self.model, WeibullAFTFitter):
                    df = pd.DataFrame(x_batch, columns=feature_names)
                    surv_df = self.model.predict_survival_function(df)
                    time_coordinates = surv_df.index.values
                    surv_prob = surv_df.values.T
                elif isinstance(self.model, ComponentwiseGradientBoostingSurvivalAnalysis):
                    pred_surv = self.model.predict_survival_function(x_batch)
                    surv_prob, time_coordinates = format_pred_sksurv(pred_surv)
                else:
                    raise ValueError("Model not supported.")

                # add 0 to time_coordinates and 1 to surv_prob if not present
                if time_coordinates[0] != 0:
                    time_coordinates = np.concatenate([np.array([0]), time_coordinates], 0)
                    surv_prob = np.concatenate([np.ones([surv_prob.shape[0], 1]), surv_prob], 1)

                time_coordinates = np.repeat(time_coordinates[np.newaxis, :], surv_prob.shape[0], axis=0)
                quantile_batch = survival_to_quantile(surv_prob, time_coordinates, quantile_levels,
                                                      self.args.interpolate)
                quantile_batchs.append(quantile_batch)
            # quantile_predictions = np.concatenate(quantile_batchs, 0)
            quantile_predictions = np.vstack(quantile_batchs)

        return quantile_predictions


class SurvivalPredictionNC(sklearn.base.BaseEstimator):
    """Nonconformity scorer using an underlying survival model."""

    def __init__(
            self,
            model,
            args=argparse.Namespace,
    ):
        super(SurvivalPredictionNC, self).__init__()
        self.model = model
        self.err_func = ProbAtEventTimeErrFunc()
        self.args = args

    def fit(self, train_set: pd.DataFrame, val_set: pd.DataFrame):
        """Fits the underlying model of the nonconformity scorer."""
        if (isinstance(self.model, CoxPH) or isinstance(self.model, MTLR)
                or isinstance(self.model, CQRNN) or isinstance(self.model, LogNormalNN)):
            folder = 'logs/SurvivalPredictionNC'
            # create folder if it does not exist
            if not os.path.exists(folder):
                os.makedirs(folder)
            self.model.fit(train_set, val_set, device=self.args.device, batch_size=self.args.batch_size,
                           epochs=self.args.n_epochs, lr=self.args.lr, lr_min=1e-3 * self.args.lr,
                           weight_decay=self.args.weight_decay, early_stop=self.args.early_stop,
                           fname=folder + f'/{self.model.__class__.__name__}', verbose=self.args.verbose)
        elif isinstance(self.model, DeepHitSingle) or isinstance(self.model, CoxTime):
            x_train = train_set.drop(['time', 'event'], axis=1).values
            t_train, e_train = train_set['time'].values, train_set['event'].values
            x_val = val_set.drop(['time', 'event'], axis=1).values
            t_val, e_val = val_set['time'].values, val_set['event'].values
            y_train = self.model.label_transform.transform(t_train, e_train)
            y_val = self.model.label_transform.transform(t_val, e_val)
            val = (x_val, y_val)
            val_size = x_val.shape[0]

            self.model.optimizer.set_lr(self.args.lr)
            self.model.optimizer.set('weight_decay', self.args.weight_decay)
            if self.args.early_stop:
                callbacks = [tt.callbacks.EarlyStopping()]
            else:
                callbacks = None

            self.model.fit(input=x_train, target=y_train, batch_size=self.args.batch_size, epochs=self.args.n_epochs,
                           callbacks=callbacks, verbose=self.args.verbose, val_data=val, val_batch_size=val_size)
            if isinstance(self.model, CoxTime):
                self.model.compute_baseline_hazards()
        elif isinstance(self.model, WeibullAFTFitter):
            if self.args.use_train:
                # if we choose to use the training set for calibration, we need to combine the train and val sets
                train_set = pd.concat([train_set, val_set], ignore_index=True)
            self.model.fit(train_set, duration_col='time', event_col='event')
        elif isinstance(self.model, ComponentwiseGradientBoostingSurvivalAnalysis):
            if self.args.use_train:
                # if we choose to use the training set for calibration, we need to combine the train and val sets
                train_set = pd.concat([train_set, val_set], ignore_index=True)
            x_train = train_set.drop(["time", "event"], axis=1).values
            t_train, e_train = train_set["time"].values, train_set["event"].values
            y_train = np.empty(dtype=[('cens', bool), ('time', np.float64)], shape=t_train.shape[0])
            y_train['cens'] = e_train
            y_train['time'] = t_train
            self.model.fit(x_train, y_train)

    def score(
            self,
            feature_df: pd.DataFrame,
            t: np.ndarray,
            e: np.ndarray
    ):
        x = feature_df.values
        x_names = feature_df.columns.tolist()

        surv_prob, time_coordinates = self.predict_nc(x, x_names)
        con_scores = self.err_func.apply(surv_prob, time_coordinates, t)
        con_scores_event = con_scores[e == 1]
        con_scores_cen = con_scores[e == 0]

        grid = np.linspace(0, 1, self.args.n_sample).reshape(self.args.n_sample, 1)
        con_scores_event = np.repeat(con_scores_event, self.args.n_sample)
        con_scores_cen = grid * con_scores_cen.reshape(1, -1)
        con_scores_cen = con_scores_cen.reshape(-1)
        conformal_scores = np.concatenate([con_scores_event, con_scores_cen])
        return conformal_scores

    def predict(
            self,
            x: np.ndarray,
            conformal_scores: np.ndarray,
            feature_names: list[str] = None,
            quantile_levels=None
    ):
        surv_prob, time_coordinates = self.predict_nc(x, feature_names)

        adj_percentile_levels = self.err_func.apply_inverse(conformal_scores, quantile_levels)

        adj_quan_levels = 1 - adj_percentile_levels
        quantile_predictions = survival_to_quantile(surv_prob, time_coordinates, adj_quan_levels, self.args.interpolate)
        return quantile_predictions

    def predict_nc(
            self,
            x: np.ndarray,
            feature_names: list[str] = None
    ):
        if isinstance(self.model, CQRNN):
            x = torch.from_numpy(x).float().to(self.args.device)
            quantile_predictions = self.model.predict_quantiles(x).cpu().numpy()
            q_levels = self.model.quan_levels.cpu().numpy()
            q_levels, quantile_predictions = make_mono_quantiles(q_levels, quantile_predictions,
                                                                 method=self.args.mono_method, seed=self.args.seed)
            time_coordinates = quantile_predictions
            surv_prob = np.repeat(1 - q_levels[np.newaxis, :], x.shape[0], axis=0)
        else:
            if isinstance(self.model, CoxPH) or isinstance(self.model, MTLR) or isinstance(self.model, LogNormalNN):
                x = torch.from_numpy(x).float().to(self.args.device)
                surv_prob = self.model.predict_survival(x).cpu().numpy()
                time_coordinates = self.model.time_bins
                if isinstance(self.model, MTLR):
                    time_coordinates = pad_tensor(time_coordinates, 0, where='start')
                time_coordinates = time_coordinates.cpu().numpy()
            elif isinstance(self.model, DeepHitSingle) or isinstance(self.model, CoxTime):
                surv_df = self.model.predict_surv_df(x)
                time_coordinates = surv_df.index.values
                surv_prob = surv_df.values.T
            elif isinstance(self.model, WeibullAFTFitter):
                df = pd.DataFrame(x, columns=feature_names)
                surv_df = self.model.predict_survival_function(df)
                time_coordinates = surv_df.index.values
                surv_prob = surv_df.values.T
            elif isinstance(self.model, ComponentwiseGradientBoostingSurvivalAnalysis):
                pred_surv = self.model.predict_survival_function(x)
                surv_prob, time_coordinates = format_pred_sksurv(pred_surv)
            else:
                raise ValueError("Model not supported.")

            # add 0 to time_coordinates and 1 to surv_prob if not present
            if time_coordinates[0] != 0:
                time_coordinates = np.concatenate([np.array([0]), time_coordinates], 0)
                surv_prob = np.concatenate([np.ones([surv_prob.shape[0], 1]), surv_prob], 1)

            time_coordinates = np.repeat(time_coordinates[np.newaxis, :], surv_prob.shape[0], axis=0)
        return surv_prob, time_coordinates
