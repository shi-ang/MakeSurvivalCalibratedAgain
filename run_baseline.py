import argparse
import os
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import trange
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector, ColumnTransformer
import wandb
import torchtuples as tt

# models
from lifelines import KaplanMeierFitter
from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis
from lifelines.fitters.weibull_aft_fitter import WeibullAFTFitter
from model import CoxPH, MTLR, CQRNN, LogNormalNN
from pycox.models import DeepHitSingle, CoxTime
from pycox.models.cox_time import MLPVanillaCoxTime

from utils import save_params, set_seed, print_performance, pad_tensor
from utils.util_survival import survival_data_split, xcal_from_hist, make_time_bins, format_pred_sksurv, \
    make_mono_quantiles
from args import generate_parser
from data import make_survival_data
from data.cond_features import get_cond_functions
from SurvivalEVAL import SurvivalEvaluator, QuantileRegEvaluator
from CondCalEvaluation import cond_xcal, wsc_xcal

folder = 'logs/Baseline'
# create folder if it does not exist
if not os.path.exists(folder):
    os.makedirs(folder)


def main(args=None):
    if isinstance(args, argparse.Namespace):
        wandb.init(
            project="conditionalCSD",
            config=args,
            name=args.model + "_" + args.data + "_baseline"
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
    data = make_survival_data(args.data)
    conditions = get_cond_functions(args.data)
    features = data.columns.to_list()
    assert "time" in data.columns and "event" in data.columns, "The event time variable and censor indicator " \
                                                               "variable is missing or need to be renamed."
    # Continuous features: median imputation + scaling
    enc_con = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
    ])
    sel_con = make_column_selector(pattern='^num_')

    # Non-continuous features: mode imputation only
    enc_cat = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='most_frequent')),
    ])
    sel_cat = make_column_selector(
        pattern='^(?!num_|time$|event$).*'  # everything NOT num_, time, or event
    )

    enc_df = ColumnTransformer(
        transformers=[
            ('num', enc_con, sel_con),
            ('cat', enc_cat, sel_cat),
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    )
    enc_df.set_output(transform='pandas')

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
        data_train = enc_df.fit_transform(data_train).astype('float32')
        data_val = enc_df.transform(data_val).astype('float32') if not data_val.empty else data_val
        data_test = enc_df.transform(data_test).astype('float32')
        data_train_val = pd.concat([data_train, data_val], ignore_index=True) if not data_val.empty else data_train

        x_train = data_train.drop(["time", "event"], axis=1).values
        t_train, e_train = data_train["time"].values, data_train["event"].values
        x_val = data_val.drop(["time", "event"], axis=1).values if not data_val.empty else None
        t_val, e_val = data_val["time"].values, data_val["event"].values if not data_val.empty else None
        x_test = data_test.drop(['time', 'event'], axis=1).values
        t_test, e_test = data_test["time"].values, data_test["event"].values
        x_train_val = data_train_val.drop(["time", "event"], axis=1).values
        t_train_val, e_train_val = data_train_val["time"].values, data_train_val["event"].values

        # create time bins for discrete survival analysis models
        if args.model in ["MTLR", "DeepHit"]:
            discrete_bins = make_time_bins(t_train, event=e_train)
            if args.model == "DeepHit":
                # the first bin of DeepHit must smaller than the smallest time in the data
                discrete_bins[0] = max(t_train_val.min() - 1e-5, 0)

        if args.model == "KM":
            model = KaplanMeierFitter()
            start_time = datetime.now()
            model.fit(t_train_val, event_observed=e_train_val)
            mid_time = datetime.now()
            km_curve = model.survival_function_.KM_estimate.values
            time_coordinates = model.survival_function_.index.values
            infer_time = (datetime.now() - mid_time).total_seconds()
            train_time = (mid_time - start_time).total_seconds()
            # use the KM curve for the training data as the prediction
            surv_test = np.repeat(km_curve[np.newaxis, :], x_test.shape[0], axis=0)
        elif args.model == "CoxPH":
            model = CoxPH(
                n_features=args.n_features,
                hidden_size=args.neurons,
                norm=args.norm,
                activation=args.activation,
                dropout=args.dropout
            )
            start_time = datetime.now()
            model.fit(data_train, data_val, device=device, batch_size=args.batch_size, epochs=args.n_epochs,
                      lr=args.lr, lr_min=1e-3 * args.lr, weight_decay=args.weight_decay, early_stop=args.early_stop,
                      fname=folder + f'/{model.__class__.__name__}', verbose=args.verbose)
            mid_time = datetime.now()
            x_test = torch.from_numpy(x_test).float().to(device)
            surv_test = model.predict_survival(x_test)
            time_coordinates = model.time_bins
            infer_time = (datetime.now() - mid_time).total_seconds()
            train_time = (mid_time - start_time).total_seconds()
        elif args.model == "MTLR":
            model = MTLR(
                n_features=args.n_features,
                time_bins=discrete_bins,
                hidden_size=args.neurons,
                norm=args.norm,
                activation=args.activation,
                dropout=args.dropout
            )
            start_time = datetime.now()
            model.fit(data_train, data_val, device=device, batch_size=args.batch_size, epochs=args.n_epochs,
                      lr=args.lr, lr_min=1e-3 * args.lr, weight_decay=args.weight_decay, early_stop=args.early_stop,
                      fname=folder + f'/{model.__class__.__name__}', verbose=args.verbose)
            mid_time = datetime.now()
            x_test = torch.from_numpy(x_test).float().to(device)
            surv_test = model.predict_survival(x_test)
            time_coordinates = model.time_bins
            time_coordinates = pad_tensor(time_coordinates, 0, where='start')
            infer_time = (datetime.now() - mid_time).total_seconds()
            train_time = (mid_time - start_time).total_seconds()
        elif args.model == "CQRNN":
            model = CQRNN(
                n_features=args.n_features,
                hidden_size=args.neurons,
                n_quantiles=args.n_quantiles,
                norm=args.norm,
                activation=args.activation,
                dropout=args.dropout
            )
            start_time = datetime.now()
            model.fit(data_train, data_val, device=device, batch_size=args.batch_size, epochs=args.n_epochs,
                      lr=args.lr, lr_min=1e-3 * args.lr, weight_decay=args.weight_decay, early_stop=args.early_stop,
                      fname=folder + f'/{model.__class__.__name__}', verbose=args.verbose)
            mid_time = datetime.now()
            x_test = torch.from_numpy(x_test).float().to(device)
            quan_test = model.predict_quantiles(x_test)
            # quan_test = pad_tensor(quan_test, 0, where='start')     # for quantile = 0, the prediction is 0
            quan_levels = model.quan_levels
            # quan_levels = pad_tensor(quan_levels, 0, where='start')
            quan_levels, quan_test = make_mono_quantiles(quan_levels.cpu().numpy(), quan_test.cpu().numpy(),
                                                         method=args.mono_method, seed=seed_i)
            infer_time = (datetime.now() - mid_time).total_seconds()
            train_time = (mid_time - start_time).total_seconds()
        elif args.model == "LogNormalNN":
            model = LogNormalNN(
                n_features=args.n_features,
                hidden_size=args.neurons,
                norm=args.norm,
                activation=args.activation,
                dropout=args.dropout,
                lam=args.lam
            )
            start_time = datetime.now()
            model.fit(data_train, data_val, device=device, batch_size=args.batch_size, epochs=args.n_epochs,
                      lr=args.lr, lr_min=1e-3 * args.lr, weight_decay=args.weight_decay, early_stop=args.early_stop,
                      fname=folder + f'/{model.__class__.__name__}', verbose=args.verbose)
            mid_time = datetime.now()
            x_test = torch.from_numpy(x_test).float().to(device)
            surv_test = model.predict_survival(x_test)
            time_coordinates = model.time_bins
            infer_time = (datetime.now() - mid_time).total_seconds()
            train_time = (mid_time - start_time).total_seconds()
        elif args.model == "DeepHit":
            labtrans = DeepHitSingle.label_transform(discrete_bins.numpy())
            net = tt.practical.MLPVanilla(in_features=args.n_features, num_nodes=args.neurons,
                                          out_features=labtrans.out_features, batch_norm=args.norm,
                                          dropout=args.dropout, activation=getattr(nn, args.activation))
            model = DeepHitSingle(net, tt.optim.Adam, device=args.device, alpha=0.2, sigma=0.1, duration_index=labtrans.cuts)
            model.label_transform = labtrans

            y_train = model.label_transform.transform(*(t_train, e_train))
            y_val = model.label_transform.transform(*(t_val, e_val))

            val = (x_val, y_val)
            val_size = x_val.shape[0]

            model.optimizer.set_lr(args.lr)
            model.optimizer.set('weight_decay', args.weight_decay)
            if args.early_stop:
                callbacks = [tt.callbacks.EarlyStopping()]
            else:
                callbacks = None
            start_time = datetime.now()
            model.fit(input=x_train, target=y_train, batch_size=args.batch_size, epochs=args.n_epochs,
                      callbacks=callbacks, verbose=args.verbose, val_data=val, val_batch_size=val_size)
            mid_time = datetime.now()
            surv_df = model.predict_surv_df(x_test)
            time_coordinates = surv_df.index.values
            surv_test = surv_df.values.T
            infer_time = (datetime.now() - mid_time).total_seconds()
            train_time = (mid_time - start_time).total_seconds()
        elif args.model == "CoxTime":
            labtrans = CoxTime.label_transform()
            labtrans.fit(t_train, e_train)
            net = MLPVanillaCoxTime(in_features=args.n_features, num_nodes=args.neurons, batch_norm=args.norm,
                                    dropout=args.dropout, activation=getattr(nn, args.activation))
            model = CoxTime(net, tt.optim.Adam, device=args.device, labtrans=labtrans)
            model.label_transform = labtrans

            y_train = model.label_transform.fit_transform(*(t_train, e_train))
            y_val = model.label_transform.transform(*(t_val, e_val))

            val = (x_val, y_val)
            val_size = x_val.shape[0]

            model.optimizer.set_lr(args.lr)
            model.optimizer.set('weight_decay', args.weight_decay)
            if args.early_stop:
                callbacks = [tt.callbacks.EarlyStopping()]
            else:
                callbacks = None
            start_time = datetime.now()
            model.fit(input=x_train, target=y_train, batch_size=args.batch_size, epochs=args.n_epochs,
                      callbacks=callbacks, verbose=args.verbose, val_data=val, val_batch_size=val_size)
            model.compute_baseline_hazards()
            mid_time = datetime.now()
            surv_df = model.predict_surv_df(x_test)
            time_coordinates = surv_df.index.values
            surv_test = surv_df.values.T

            # add the initial time point
            time_coordinates = np.concatenate([np.array([0]), time_coordinates], 0)
            surv_test = np.concatenate([np.ones([surv_test.shape[0], 1]), surv_test], 1)
            infer_time = (datetime.now() - mid_time).total_seconds()
            train_time = (mid_time - start_time).total_seconds()
        elif args.model == "GB":
            y_train_val = np.empty(dtype=[('cens', bool), ('time', np.float64)], shape=t_train_val.shape[0])
            y_train_val['cens'] = e_train_val
            y_train_val['time'] = t_train_val

            model = ComponentwiseGradientBoostingSurvivalAnalysis(loss='coxph', n_estimators=100, random_state=seed_i)
            start_time = datetime.now()
            model.fit(x_train_val, y_train_val)
            mid_time = datetime.now()
            pred_surv = model.predict_survival_function(x_test)
            surv_test, time_coordinates = format_pred_sksurv(pred_surv)
            infer_time = (datetime.now() - mid_time).total_seconds()
            train_time = (mid_time - start_time).total_seconds()
        elif args.model == "AFT":
            model = WeibullAFTFitter(penalizer=0.01)
            start_time = datetime.now()
            model.fit(data_train_val, duration_col='time', event_col='event')
            mid_time = datetime.now()
            surv_df = model.predict_survival_function(data_test)
            time_coordinates = surv_df.index.values
            surv_test = surv_df.values.T
            time_coordinates = np.concatenate([np.array([0]), time_coordinates], 0)
            surv_test = np.concatenate([np.ones([surv_test.shape[0], 1]), surv_test], 1)
            infer_time = (datetime.now() - mid_time).total_seconds()
            train_time = (mid_time - start_time).total_seconds()
        else:
            raise ValueError(f"Unknown model name: {args.model}")

        # evaluate the performance
        if args.model != "CQRNN":
            evaler = SurvivalEvaluator(surv_test, time_coordinates, t_test, e_test, t_train_val, e_train_val,
                                       predict_time_method="Median", interpolation='Pchip')
        else:
            evaler = QuantileRegEvaluator(quan_test, quan_levels, t_test, e_test, t_train_val, e_train_val,
                                          predict_time_method="Median", interpolation='Pchip')
        c_index = evaler.concordance(ties="All")[0]
        ibs_score = evaler.integrated_brier_score(num_points=10)
        hinge_abs = evaler.mae(method='Hinge', verbose=False, weighted=True)
        po_abs = evaler.mae(method='Pseudo_obs', verbose=False, weighted=False)
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
                   'infer_time': infer_time})
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
        cal_times=train_times,
        infer_times=infer_times
    )


if __name__ == '__main__':
    # enable for debugging
    # torch.autograd.set_detect_anomaly(True)

    args = generate_parser()
    main(args)
    wandb.finish()
