from abc import abstractmethod
from tqdm import trange
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from model.loss import LikelihoodMTLR, PartialLikelihood, CensoredPinballLoss, LikelihoodLogNormal, CRPS
from utils.util_survival import baseline_hazard, extract_survival


def build_sequential_nn(in_features, hidden_dims, batch_norm, activation, dropout):
    """Build a sequential neural network, except last layer."""
    layers = []
    for i in range(len(hidden_dims)):
        if i == 0:
            layers.append(nn.Linear(in_features, hidden_dims[i]))
        else:
            layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))

        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dims[i]))

        layers.append(getattr(nn, activation)())

        if dropout is not None:
            layers.append(nn.Dropout(dropout))
    return layers


class BaseMTLR(nn.Module):
    """Multi-task logistic regression for individualised
    survival prediction.

    The MTLR time-logits are computed as:
    `z = sum_k x^T w_k + b_k`,
    where `w_k` and `b_k` are learnable weights and biases for each time
    interval.

    Note that a slightly more efficient reformulation is used here, first
    proposed in [2]_.

    References
    ----------
    ..[1] C.-N. Yu et al., ‘Learning patient-specific cancer survival
    distributions as a sequence of dependent regressors’, in Advances in neural
    information processing systems 24, 2011, pp. 1845–1853.
    ..[2] P. Jin, ‘Using Survival Prediction Techniques to Learn
    Consumer-Specific Reservation Price Distributions’, Master's thesis,
    University of Alberta, Edmonton, AB, 2015.
    """

    def __init__(self, in_features: int, num_time_bins: int):
        """Initialises the module.

        Parameters
        ----------
        in_features
            Number of input features.
        num_time_bins
            The number of bins to divide the time axis into.
        """
        super().__init__()
        if num_time_bins < 1:
            raise ValueError("The number of time bins must be at least 1")
        if in_features < 1:
            raise ValueError("The number of input features must be at least 1")
        self.in_features = in_features
        self.num_time_bins = num_time_bins + 1  # + extra time bin [max_time, inf)

        self.mtlr_weight = nn.Parameter(torch.Tensor(self.num_time_bins - 1, self.in_features))
        self.mtlr_bias = nn.Parameter(torch.Tensor(self.num_time_bins - 1))

        # `G` is the coding matrix from [2]_ used for fast summation.
        # When registered as buffer, it will be automatically
        # moved to the correct device and stored in saved
        # model state.
        self.register_buffer(
            "G",
            torch.tril(
                torch.ones(self.num_time_bins - 1,
                           self.num_time_bins,
                           requires_grad=True)))
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass on a batch of examples.

        Parameters
        ----------
        x : torch.Tensor, shape (num_samples, num_features)
            The input data.

        Returns
        -------
        torch.Tensor, shape (num_samples, num_time_bins - 1)
            The predicted time logits.
        """
        out = F.linear(x, self.mtlr_weight, self.mtlr_bias)
        return torch.matmul(out, self.G)

    def reset_parameters(self):
        """Resets the model parameters."""
        nn.init.xavier_normal_(self.mtlr_weight)
        nn.init.constant_(self.mtlr_bias, 0.)

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.in_features}, num_time_bins={self.num_time_bins})"


class SurvivalBase(nn.Module):
    """Base class for survival models."""

    def __init__(self, n_features: int, hidden_size: list, norm: bool, activation: str, dropout: float):
        super().__init__()
        self.in_features = n_features
        self.output_size = None
        self.dropout = dropout
        self.norm = norm
        self.hidden_size = hidden_size
        self.activation = activation
        self.loss = None
        self.model = None

    def forward(self, x):
        return self.model(x)

    @abstractmethod
    def predict_survival(self, x):
        pass

    def predict_cdf(self, x):
        return 1 - self.predict_survival(x)

    def predict_time(self, x, pred_type='mean'):
        survival = self.predict_survival(x)
        if pred_type == 'mean':
            # integral the survival function
            return torch.trapz(survival, torch.cat([torch.tensor([0]), self.time_bins], dim=0).to(survival.device), dim=1)
        elif pred_type == 'median':
            # fit the survival function using spline, then find the median
            raise NotImplementedError
        else:
            raise ValueError("pred_type should be either 'mean' or 'median'")

    def fit(
            self,
            train_df: pd.DataFrame,
            val_df: pd.DataFrame,
            device: torch.device,
            batch_size: int,
            epochs: int,
            lr: float,
            lr_min: float,
            weight_decay: float,
            early_stop: bool = True,
            patience: int = 50,
            fname: str = '',
            verbose: bool = True
    ):
        self.reset_parameters()
        self.to(device)

        optimizer = torch.optim.Adam((param for param in self.parameters() if param.requires_grad), lr=lr,
                                     weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_min)
        x_train, y_train, _, _ = extract_survival(train_df, self.time_bins)
        train_dataloader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

        if not val_df.empty:
            x_val, y_val, _, _ = extract_survival(val_df, self.time_bins)
            x_val, y_val = x_val.to(device), y_val.to(device)

        best_loss = float('inf')
        best_ep = -1

        # training and evaluation
        prefix = f'Training w Early Stop on {device}' if early_stop else f'Training on {device} w/o Early Stop'
        pbar = trange(epochs, disable=not verbose, desc=prefix)
        for ep in pbar:
            # start training
            self.train()
            train_loss_ep = 0
            for xi, yi in train_dataloader:
                xi, yi = xi.to(device), yi.to(device)
                y_pred = self(xi)

                loss = self.loss(y_pred, yi)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_ep += loss.detach().item()

            scheduler.step()
            train_loss_ep /= len(train_dataloader)
            # evaluation
            self.eval()
            with torch.no_grad():
                postfix = f"Train loss = {train_loss_ep:.4f};"

                if early_stop and not val_df.empty:
                    y_val_pred = self(x_val)
                    eval_loss = self.loss(y_val_pred, y_val)
                    postfix += f" Val loss = {eval_loss:.4f};"

                    if best_loss > eval_loss:
                        best_loss = eval_loss
                        best_ep = ep
                        torch.save({'model_state_dict': self.state_dict()}, fname + '.pth')
                    if (ep - best_ep) > patience:
                        postfix += f" Early stop at epoch {ep}. Best epoch is {best_ep}. Start testing..."
                        pbar.set_postfix_str(postfix)
                        break
                pbar.set_postfix_str(postfix)
        self.load_state_dict(torch.load(fname + '.pth')['model_state_dict']) if early_stop and not val_df.empty else None

    def reset_parameters(self):
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


class MTLR(SurvivalBase):
    """MTLR model with regularization"""

    def __init__(self, n_features: int, time_bins: torch.Tensor, hidden_size: list, norm: bool, activation: str,
                 dropout: float):
        super(MTLR, self).__init__(n_features, hidden_size, norm, activation, dropout)
        output_size = len(time_bins)
        self.time_bins = time_bins
        self.output_size = output_size
        self.loss = LikelihoodMTLR(reduction='mean')

        self.model = self._build_model()

    def _build_model(self):
        if not self.hidden_size:
            layers = [BaseMTLR(self.in_features, self.output_size)]
        else:
            layers = build_sequential_nn(self.in_features, self.hidden_size, self.norm, self.activation, self.dropout)
            layers.append(BaseMTLR(self.hidden_size[-1], self.output_size))
        return nn.Sequential(*layers)

    def predict_survival(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            assert logits.dim() == 2, "The logits should have dimension with with size (n_data, n_bins)"
            G = torch.tril(torch.ones(logits.shape[1], logits.shape[1])).to(logits.device)
            density = torch.softmax(logits, dim=1)
            return torch.matmul(density, G)


class CoxPH(SurvivalBase):
    """CoxPH model with regularization"""
    def __init__(self, n_features: int, hidden_size: list, norm: bool, activation: str, dropout: float):
        super(CoxPH, self).__init__(n_features, hidden_size, norm, activation, dropout)
        self.output_size = 1
        self.time_bins = None
        self.baseline_hazard = None
        self.cum_baseline_hazard = None
        self.baseline_survival = None
        self.loss = PartialLikelihood(reduction='mean')

        self.model = self._build_model()

    def _build_model(self):
        if not self.hidden_size:
            # if hidden_size is empty, then the only layer is linear
            layers = [nn.Linear(self.in_features, self.output_size)]
        else:
            layers = build_sequential_nn(self.in_features, self.hidden_size, self.norm, self.activation, self.dropout)
            layers.append(nn.Linear(self.hidden_size[-1], self.output_size))
        return nn.Sequential(*layers)

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame, device: torch.device, batch_size: int,
            epochs: int, lr: float, lr_min: float, weight_decay: float, early_stop: bool = True,
            patience: int = 50, fname: str = '', verbose: bool = True):
        super(CoxPH, self).fit(train_df, val_df, device, batch_size, epochs, lr, lr_min, weight_decay,
                               early_stop, patience, fname, verbose)
        self.cal_baseline_survival(train_df)

    def predict_risk(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def predict_survival(self, x):
        self.eval()
        with torch.no_grad():
            risks = self.predict_risk(x)
            n_data = len(risks)
            risk_score = torch.exp(risks)
            risk_score = risk_score.squeeze()
            survival_curves = torch.empty((n_data, self.baseline_survival.shape[0]), dtype=torch.double).to(
                risks.device)
            for i in range(n_data):
                survival_curve = torch.pow(self.baseline_survival, risk_score[i])
                survival_curves[i] = survival_curve
            return survival_curves

    def cal_baseline_survival(self, dataset):
        x, _, t, e = extract_survival(dataset)
        device = next(self.parameters()).device
        x, t, e = x.to(device), t.to(device), e.to(device)
        with torch.no_grad():
            outputs = self.forward(x)
        self.time_bins, self.baseline_hazard, self.cum_baseline_hazard, self.baseline_survival = baseline_hazard(outputs, t, e)


class CQRNN(nn.Module):
    """Simple MLP model, use for CQRNN model"""
    def __init__(self, n_features: int, hidden_size: list, n_quantiles: int,
                 norm: bool, activation: str, dropout: float):
        super(CQRNN, self).__init__()
        self.in_features = n_features
        if n_quantiles is None:
            self.n_quantiles = 9
        else:
            assert isinstance(n_quantiles, int) and n_quantiles > 0, "n_quantiles must be a positive integer"
            self.n_quantiles = n_quantiles
        self.dropout = dropout
        self.norm = norm
        self.hidden_size = hidden_size
        self.activation = activation
        self.quan_levels = torch.linspace(1 / (self.n_quantiles + 1), self.n_quantiles / (self.n_quantiles + 1),
                                          self.n_quantiles, dtype=torch.float64)
        self.loss = CensoredPinballLoss(self.quan_levels, use_cross_loss=True, reduction='sum')

        self.model = self._build_model()

    def _build_model(self):
        if not self.hidden_size:
            layers = [nn.Linear(self.in_features, self.n_quantiles)]
        else:
            layers = build_sequential_nn(self.in_features, self.hidden_size, self.norm, self.activation, self.dropout)
            layers.append(nn.Linear(self.hidden_size[-1], self.n_quantiles))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def fit(
            self,
            train_df: pd.DataFrame,
            val_df: pd.DataFrame,
            device: torch.device,
            batch_size: int,
            epochs: int,
            lr: float,
            lr_min: float,
            weight_decay: float,
            early_stop: bool = True,
            patience: int = 50,
            fname: str = '',
            verbose: bool = True
    ):
        self.reset_parameters()
        self.to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_min)
        x_train, y_train, t_train, _ = extract_survival(train_df, None)
        train_dataloader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
        n_train = x_train.shape[0]
        self.loss.t_max = 1.2 * t_train.max()

        if not val_df.empty:
            x_val, y_val, _, _ = extract_survival(val_df, None)
            x_val, y_val = x_val.to(device), y_val.to(device)
            n_val = x_val.shape[0]

        best_loss = float('inf')
        best_ep = -1

        # training and evaluation
        prefix = f'Training w Early Stop on {device}' if early_stop else f'Training on {device} w/o Early Stop'
        pbar = trange(epochs, disable=not verbose, desc=prefix)
        for ep in pbar:
            self.train()
            train_loss_ep = 0.
            for xi, yi in train_dataloader:
                xi, yi = xi.to(device), yi.to(device)
                y_pred = self(xi)

                loss = self.loss(y_pred, yi)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_ep += loss.detach().item()

            scheduler.step()
            train_loss_ep /= n_train
            # evaluation
            self.eval()
            with torch.no_grad():
                postfix = f"Train loss = {train_loss_ep:.4f};"

                if early_stop and not val_df.empty:
                    y_val_pred = self(x_val)
                    eval_loss = self.loss(y_val_pred, y_val)
                    postfix += f" Val loss = {eval_loss / n_val:.4f};"

                    if best_loss > eval_loss:
                        best_loss = eval_loss
                        best_ep = ep
                        torch.save({'model_state_dict': self.state_dict()}, fname + '.pth')
                    if (ep - best_ep) > patience:
                        postfix += f" Early stop at epoch {ep}. Best epoch is {best_ep}. Start testing..."
                        pbar.set_postfix_str(postfix)
                        break
                pbar.set_postfix_str(postfix)
        self.load_state_dict(torch.load(fname + '.pth')['model_state_dict']) if early_stop and not val_df.empty else None

    def predict_quantiles(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def reset_parameters(self):
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


class LogNormalNN(nn.Module):
    def __init__(self, n_features: int, hidden_size: list, norm: bool, activation: str, dropout: float,
                 lam: float = 0.0):
        super(LogNormalNN, self).__init__()
        self.in_features = n_features
        self.dropout = dropout
        self.norm = norm
        self.hidden_size = hidden_size
        self.activation = activation
        self.output_size = 2
        self.t_max = None
        self.time_bins = None
        self.time_bin_before_rescale = None
        self.rescale = None
        self.loss = LikelihoodLogNormal(reduction="sum", lam=lam, type='sfm') # type='sfm' or 'x-cal'
        # self.loss = CRPS()

        if not self.hidden_size:
            self.mu_model = nn.Linear(self.in_features, 1)
            self.sigma_model = nn.Linear(self.in_features, 1)
        else:
            mu_model = build_sequential_nn(n_features, hidden_size, norm, activation, dropout)
            sigma_model = build_sequential_nn(n_features, hidden_size, norm, activation, dropout)
            mu_model.append(nn.Linear(hidden_size[-1], 1))
            sigma_model.append(nn.Linear(hidden_size[-1], 1))
            self.mu_model = nn.Sequential(*mu_model)
            self.sigma_model = nn.Sequential(*sigma_model)

    def forward(self, x):
        mu = self.mu_model(x).view(-1, 1)
        pre_log_sigma = self.sigma_model(x).view(-1, 1)
        pred = torch.cat([mu, pre_log_sigma], dim=1)
        return pred

    def fit(
            self,
            train_df: pd.DataFrame,
            val_df: pd.DataFrame,
            device: torch.device,
            batch_size: int,
            epochs: int,
            lr: float,
            lr_min: float,
            weight_decay: float,
            early_stop: bool = True,
            patience: int = 50,
            fname: str = '',
            verbose: bool = True
    ):
        self.reset_parameters()
        self.to(device)

        new_train = train_df.copy()
        self.t_max = new_train.time.max()
        self.rescale = self.t_max / 1
        new_train.time = new_train.time / self.rescale

        optimizer = torch.optim.Adam((param for param in self.parameters() if param.requires_grad), lr=lr,
                                     weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_min)
        x_train, y_train, t_train, _ = extract_survival(new_train, self.time_bins)
        train_dataloader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

        if not val_df.empty:
            x_val, y_val, _, _ = extract_survival(val_df, self.time_bins)
            x_val, y_val = x_val.to(device), y_val.to(device)

        self.time_bin_before_rescale = torch.linspace(0, 5, 1000).to(device)
        self.time_bins = torch.linspace(0, self.t_max, 1000).to(device)

        best_loss = float('inf')
        best_ep = -1

        # training and evaluation
        prefix = f'Training w Early Stop on {device}' if early_stop else f'Training on {device} w/o Early Stop'
        pbar = trange(epochs, disable=not verbose, desc=prefix)
        for ep in pbar:
            # start training
            self.train()
            train_loss_ep = 0
            for xi, yi in train_dataloader:
                xi, yi = xi.to(device), yi.to(device)
                y_pred = self(xi)

                loss = self.loss(y_pred, yi)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_ep += loss.detach().item()

            scheduler.step()
            train_loss_ep /= len(train_dataloader)
            # evaluation
            self.eval()
            with torch.no_grad():
                postfix = f"Train loss = {train_loss_ep:.4f};"

                if early_stop and not val_df.empty:
                    y_val_pred = self(x_val)
                    eval_loss = self.loss(y_val_pred, y_val)
                    postfix += f" Val loss = {eval_loss:.4f};"

                    if best_loss > eval_loss:
                        best_loss = eval_loss
                        best_ep = ep
                        torch.save({'model_state_dict': self.state_dict()}, fname + '.pth')
                    if (ep - best_ep) > patience:
                        postfix += f" Early stop at epoch {ep}. Best epoch is {best_ep}. Start testing..."
                        pbar.set_postfix_str(postfix)
                        break
                pbar.set_postfix_str(postfix)
        self.load_state_dict(torch.load(fname + '.pth')['model_state_dict']) if early_stop and not val_df.empty else None

    def predict_survival(self, x):
        self.eval()
        with torch.no_grad():
            pred_params = self.forward(x)
            mu = pred_params[:, 0]
            pre_log_sigma = pred_params[:, 1]
            log_sigma = F.softplus(pre_log_sigma) - 0.5
            sigma = log_sigma.clamp(max=10).exp()
            sigma = sigma + 1e-8
            # get the survival function for each pair of mu and sigma
            cdf = torch.distributions.LogNormal(mu, sigma).cdf(self.time_bin_before_rescale.repeat(mu.shape[0], 1).T).T
            survival = 1 - cdf
            return survival

    def reset_parameters(self):
        for layer in self.mu_model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.sigma_model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
