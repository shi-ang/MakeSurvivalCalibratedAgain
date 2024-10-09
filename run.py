import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import trange
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import wandb
import torchtuples as tt

# models
from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis
from lifelines.fitters.weibull_aft_fitter import WeibullAFTFitter
from model import CoxPH, MTLR, CQRNN, LogNormalNN
from pycox.models import DeepHitSingle, CoxTime
from pycox.models.cox_time import MLPVanillaCoxTime

# Conformality
from icp import ConformalSurvDist, CSDiPOT
from icp.scorer import QuantileRegressionNC, SurvivalPredictionNC

from utils import save_params, set_seed, print_performance
from utils.util_survival import survival_data_split, xcal_from_hist, make_time_bins
from args import generate_parser
from data import make_survival_data
from data.cond_features import get_cond_functions
from SurvivalEVAL import QuantileRegEvaluator
from CondCalEvaluation import cond_xcal, wsc_xcal


def main(args=None):
    if isinstance(args, argparse.Namespace):
        wandb.init(
            project="conditionalCSD",
            config=args,
            name=args.model + "_" + args.data
        )
    else:
        wandb.init(config=args)
    wandb.define_metric("C-index", summary="mean")
    wandb.define_metric("IBS", summary="mean")
    wandb.define_metric("MAE_Hinge", summary="mean")
    wandb.define_metric("MAE_PO", summary="mean")
    wandb.define_metric("KM-cal", summary="mean")
    wandb.define_metric("X-cal", summary="mean")
    wandb.define_metric("cond_xcal", summary="mean")
    wandb.define_metric("wsc_xcal", summary="mean")
    wandb.define_metric("train_time", summary="mean")
    wandb.define_metric("infer_time", summary="mean")

    args = wandb.config
    data, cols_stdz = make_survival_data(args.data)
    conditions = get_cond_functions(args.data)
    features = data.columns.to_list()
    assert "time" in data.columns and "event" in data.columns, "The event time variable and censor indicator " \
                                                               "variable is missing or need to be renamed."
    cols_wo_stdz = list(set(features).symmetric_difference(cols_stdz))  # including time and event
    stdz = [([col], StandardScaler()) for col in cols_stdz]
    wo_stdz = [(col, None) for col in cols_wo_stdz]
    columns_transform = stdz + wo_stdz
    if args.early_stop:
        pct_train = 0.8
        pct_val = 0.1
        pct_test = 0.1
    else:
        pct_train = 0.9
        pct_val = 0.0
        pct_test = 0.1

    args.n_features = len(features) - 2     # excluding time and event
    args.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    path = save_params(args)

    ci = []
    mae_hinge = []
    mae_po = []
    ibs = []
    km_cal = []
    xcal_stats = []
    cond_xcal_stats = []
    wsc_xcal_stats = []
    train_times = []
    infer_times = []

    pbar_outer = trange(args.n_exp, disable=not args.verbose, desc='Experiment')
    for i in pbar_outer:
        seed_i = args.seed + i
        set_seed(seed_i, device)
        data_train, data_val, data_test = survival_data_split(data, stratify_colname='both', frac_train=pct_train,
                                                              frac_val=pct_val, frac_test=pct_test, random_state=seed_i)
        # standardize the data
        # [features] to keep the order, otherwise the feature order will be changed and the result is not reproducible
        mapper_df = DataFrameMapper(columns_transform, df_out=True)
        data_train = mapper_df.fit_transform(data_train).astype('float32')[features]
        data_val = mapper_df.transform(data_val).astype('float32')[features] if not data_val.empty else data_val
        data_test = mapper_df.transform(data_test).astype('float32')[features]

        # get the labels for evaluation
        t_train, e_train = data_train["time"].values, data_train["event"].values
        t_val, e_val = data_val["time"].values, data_val["event"].values if not data_val.empty else None
        x_test = data_test.drop(['time', 'event'], axis=1).values
        t_test, e_test = data_test["time"].values, data_test["event"].values
        t_train_val = np.concatenate((t_train, t_val)) if not data_val.empty else t_train
        e_train_val = np.concatenate((e_train, e_val)) if not data_val.empty else e_train

        # this is make sure MTLR and DeepHit have the same number of bins, but the bins locations are different
        # -- MTLR uses the uniformly-divided quantiles, while DeepHit uses the uniformly-divided times.
        if args.model in ["MTLR", "DeepHit"]:
            discrete_bins = make_time_bins(t_train, event=e_train)
            if args.model == "DeepHit":
                # the first bin of DeepHit must be smaller than the smallest time in the data
                discrete_bins[0] = max(t_train_val.min() - 1e-5, 0)

        if args.model == "CoxPH":
            model = CoxPH(
                n_features=args.n_features,
                hidden_size=args.neurons,
                norm=args.norm,
                activation=args.activation,
                dropout=args.dropout
            )
        elif args.model == "MTLR":
            model = MTLR(
                n_features=args.n_features,
                time_bins=discrete_bins,
                hidden_size=args.neurons,
                norm=args.norm,
                activation=args.activation,
                dropout=args.dropout
            )
        elif args.model == "CQRNN":
            model = CQRNN(
                n_features=args.n_features,
                hidden_size=args.neurons,
                n_quantiles=args.n_quantiles,
                norm=args.norm,
                activation=args.activation,
                dropout=args.dropout
            )
        elif args.model == "LogNormalNN":
            model = LogNormalNN(
                n_features=args.n_features,
                hidden_size=args.neurons,
                norm=args.norm,
                activation=args.activation,
                dropout=args.dropout,
                lam=args.lam
            )
        elif args.model == "DeepHit":
            labtrans = DeepHitSingle.label_transform(discrete_bins.numpy())
            net = tt.practical.MLPVanilla(in_features=args.n_features, num_nodes=args.neurons,
                                          out_features=labtrans.out_features, batch_norm=args.norm,
                                          dropout=args.dropout, activation=getattr(nn, args.activation))
            model = DeepHitSingle(net, tt.optim.Adam, device=args.device, alpha=0.2, sigma=0.1,
                                  duration_index=labtrans.cuts)
            model.label_transform = labtrans
        elif args.model == "CoxTime":
            labtrans = CoxTime.label_transform()
            labtrans.fit(t_train, e_train)
            net = MLPVanillaCoxTime(in_features=args.n_features, num_nodes=args.neurons, batch_norm=args.norm,
                                    dropout=args.dropout, activation=getattr(nn, args.activation))
            model = CoxTime(net, tt.optim.Adam, device=args.device, labtrans=labtrans)
            model.label_transform = labtrans
        elif args.model == "GB":
            model = ComponentwiseGradientBoostingSurvivalAnalysis(loss='coxph', n_estimators=100, random_state=seed_i)
        elif args.model == "AFT":
            model = WeibullAFTFitter(penalizer=0.01)
        else:
            raise ValueError(f"Unknown model name: {args.model}")

        if args.post_process == "CSD":
            nc_model = QuantileRegressionNC(model, args)
            icp = ConformalSurvDist(nc_model, condition=None, decensor_method=args.decensor_method,
                                    n_quantiles=args.n_quantiles)
        elif args.post_process == "CSD-iPOT":
            nc_model = SurvivalPredictionNC(model, args)
            icp = CSDiPOT(nc_model, decensor_method=args.decensor_method, n_percentile=args.n_quantiles)
        else:
            raise ValueError(f"Unknown post-processing method: {args.post_process}")

        # Fit the ICP using the proper training set, and using valset for early stopping
        start_time = datetime.now()
        icp.fit(data_train, data_val)
        # Calibrate the ICP using the calibration set
        if args.use_train:
            data_val = pd.concat([data_train, data_val], ignore_index=True)
        icp.calibrate(data_val)
        mid_time = datetime.now()
        # Produce predictions for the test set
        quan_levels, quan_preds = icp.predict(x_test)
        end_time = datetime.now()

        train_time = (mid_time - start_time).total_seconds()
        infer_time = (end_time - mid_time).total_seconds()

        # evaluate the performance
        evaler = QuantileRegEvaluator(quan_preds, quan_levels, t_test, e_test, t_train_val, e_train_val,
                                      predict_time_method="Median", interpolation='Pchip')
        c_index = evaler.concordance(ties="All")[0]
        ibs_score = evaler.integrated_brier_score(num_points=10)
        hinge_abs = evaler.mae(method='Hinge', verbose=False, weighted=True)
        po_abs = evaler.mae(method='Pseudo_obs', verbose=False, weighted=True)
        km_cal_score = evaler.km_calibration()
        _, dcal_hist = evaler.d_calibration()
        xcal_score = xcal_from_hist(dcal_hist)
        pred_probs = evaler.predict_probability_from_curve(evaler.event_times)
        cond_xcal_score = cond_xcal(x_test, e_test, pred_probs, conditions)
        if data.shape[0] >= 1000:
            wsc_xcal_score = wsc_xcal(x_test, e_test, pred_probs, random_state=seed_i)
        else:
            wsc_xcal_score = 0  # not enough data to compute the WSC

        ci.append(c_index)
        ibs.append(ibs_score)
        mae_hinge.append(hinge_abs)
        mae_po.append(po_abs)
        km_cal.append(km_cal_score)
        xcal_stats.append(xcal_score)
        cond_xcal_stats.append(cond_xcal_score)
        wsc_xcal_stats.append(wsc_xcal_score)
        train_times.append(train_time)
        infer_times.append(infer_time)

        wandb.log({'C-index': c_index,
                   'IBS': ibs_score,
                   'MAE_Hinge': hinge_abs,
                   'MAE_PO': po_abs,
                   'KM-cal': km_cal_score,
                   'X-cal': xcal_score,
                   'cond_xcal': cond_xcal_score,
                   'wsc_xcal': wsc_xcal_score,
                   'train_time': train_time,
                   'infer_time': infer_time
                   })
    print_performance(
        path=path,
        Cindex=ci,
        IBS=ibs,
        MAE_Hinge=mae_hinge,
        MAE_PO=mae_po,
        KM_cal=km_cal,
        xCal_stats=xcal_stats,
        cond_xCal_stats=cond_xcal_stats,
        wsc_xCal_stats=wsc_xcal_stats,
        train_times=train_times,
        infer_times=infer_times
    )


if __name__ == '__main__':
    # enable for debugging
    # torch.autograd.set_detect_anomaly(True)

    args = generate_parser()
    main(args)
    wandb.finish()
