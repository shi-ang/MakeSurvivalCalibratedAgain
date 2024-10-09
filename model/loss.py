import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import safe_log
from lifelines import KaplanMeierFitter


def masked_logsumexp(
        x: torch.Tensor,
        mask: torch.Tensor,
        dim: int = -1
) -> torch.Tensor:
    """Computes logsumexp over elements of a tensor specified by a mask (two-level)
    in a numerically stable way.

    Parameters
    ----------
    x
        The input tensor.
    mask
        A tensor with the same shape as `x` with 1s in positions that should
        be used for logsumexp computation and 0s everywhere else.
    dim
        The dimension of `x` over which logsumexp is computed. Default -1 uses
        the last dimension.

    Returns
    -------
    torch.Tensor
        Tensor containing the logsumexp of each row of `x` over `dim`.
    """
    max_val, _ = (x * mask).max(dim=dim)
    max_val = torch.clamp_min(max_val, 0)
    return safe_log(torch.sum(torch.exp((x - max_val.unsqueeze(dim)) * mask) * mask, dim=dim)) + max_val


class PartialLikelihood(nn.Module):
    """Computes the negative log-likelihood of a batch of model predictions."""

    def __init__(self, reduction="mean"):
        super(PartialLikelihood, self).__init__()
        assert reduction in ["mean", "sum"], "reduction must be one of 'mean', 'sum'"
        self.reduction = reduction

    def forward(self, risk_pred, y_true):
        t_true, e_true = y_true[:, 0], y_true[:, 1]
        risk_pred = risk_pred.reshape(-1, 1)
        t_true = t_true.reshape(-1, 1)
        e_true = e_true.reshape(-1, 1)
        mask = torch.ones(t_true.shape[0], t_true.shape[0]).to(t_true.device)
        mask[(t_true.T - t_true) > 0] = 0
        max_risk = risk_pred.max()
        log_loss = torch.exp(risk_pred - max_risk) * mask
        log_loss = torch.sum(log_loss, dim=0)
        log_loss = safe_log(log_loss).reshape(-1, 1) + max_risk
        # Sometimes in the batch we got all censoring data, so the denominator gets 0 and throw nan.
        # Solution: Consider increase the batch size. After all the nll should be performed on the whole dataset.
        # Based on equation 2&3 in https://arxiv.org/pdf/1606.00931.pdf
        nll = -torch.sum((risk_pred - log_loss) * e_true) / torch.sum(e_true)

        if self.reduction == "mean":
            nll = nll / risk_pred.shape[0]
        elif self.reduction == "sum":
            nll = nll

        return nll


class LikelihoodMTLR(nn.Module):
    """Computes the negative log-likelihood of a batch of model predictions."""

    def __init__(self, reduction="mean"):
        super(LikelihoodMTLR, self).__init__()
        assert reduction in ["mean", "sum"], "reduction must be one of 'mean', 'sum'"
        self.reduction = reduction

    def forward(self, logits, target_encoded):
        censored = target_encoded.sum(dim=1) > 1
        nll_censored = masked_logsumexp(logits[censored], target_encoded[censored]).sum() if censored.any() else 0
        nll_uncensored = (logits[~censored] * target_encoded[~censored]).sum() if (~censored).any() else 0

        # the normalising constant
        norm = torch.logsumexp(logits, dim=1).sum()

        nll_total = -(nll_censored + nll_uncensored - norm)
        if self.reduction == "mean":
            nll_total = nll_total / target_encoded.size(0)
        elif self.reduction == "sum":
            nll_total = nll_total

        return nll_total


def crossing_loss(y_pred):
    # crossing loss
    # y_pred is size (n_batch, n_quantiles)
    # where adjacent quantiles are consecutive
    # https://stats.stackexchange.com/questions/249874/the-issue-of-quantile-curves-crossing-each-other
    loss_cross = 0
    margin=0.1
    alpha=10
    diffs = y_pred[:, 1:] - y_pred[:, :-1] # we would like diffs all to be +ve if not crossing
    # diffs = y_pred[:,1:-1] - y_pred[:,:-2] # we would like diffs all to be +ve if not crossing
    loss_cross = alpha*torch.mean(torch.maximum(torch.tensor(0.0), margin -diffs))
    return loss_cross


def quantile_loss(y_pred, y_true, cen_indicator, taus_torch):
    # standard checkmark / tilted pinball loss used for quantile regression
    # but we also pass in cen_indicator and avoid calculating this over those datapoints

    tau_block = taus_torch.repeat((cen_indicator.shape[0], 1))  # need this stacked in shape (n_batch, n_quantiles)
    loss = torch.sum((cen_indicator < 1) * (y_pred - y_true) * ((1 - tau_block) - 1. * (y_pred < y_true)), dim=1)
    loss = torch.mean(loss)
    # I thought about whether this should be /N (mean as here), or /N_observed, torch.sum(loss)/torch.sum(cen_indicator<1)
    # and same for censored loss
    # I confirmed it definitely should all be /N, so fine to use mean
    return loss


def cqrnn_loss(y_pred, y_true, taus_torch, IS_USE_CROSS_LOSS, y_max):
    t_true, e_true = y_true[:, 0], y_true[:, 1]
    t_true = t_true.reshape(-1, 1)
    e_true = e_true.reshape(-1, 1)
    c_true = 1 - e_true
    # this is CQRNN loss as in paper
    # y_pred is shape (n_batch, n_quantiles)
    # y_true is shape (n_batch, 1)
    # cen_indicator is shape (n_batch, 1)

    # we've taken care to implement the loss without for loops, so things can be parallelised quickly
    # but the downside is that this becomes harder to read and match up with the description in the paper
    # so we also include cqrnn_loss_slowforloops()
    # just note they both do the same thing

    # 1) first do all observed data points, censored loss not required
    # 2) second do all censored observations, no observed points

    # use detach to figure out where to block
    # first figure out closest quantile (do for all observations)
    y_pred_detach = y_pred.detach()
    # do we need detach()? yes I think so, otherwise loss is affected, though it's argmin so gradients prob don't flow anyway

    # should do this outside loss really and subselect here if needed
    tau_block = taus_torch.repeat((c_true.shape[0], 1))  # need this stacked in shape (n_batch, n_quantiles),

    loss_obs = quantile_loss(y_pred, t_true, c_true, taus_torch)

    # add in crossing loss
    if IS_USE_CROSS_LOSS:
        loss_obs += crossing_loss(y_pred)

    # use argmin to get nearest quantile
    torch_abs = torch.abs(
        t_true - y_pred_detach[:, :-1])  # ignore the final quantile, which represents 1.0, so use [:-1]
    estimated_quantiles = torch.max(
        tau_block[:, :-1] * (torch_abs == torch.min(torch_abs, dim=1).values.view(torch_abs.shape[0], 1)), dim=1).values

    # compute weights, eq 11, portnoy 2003
    # want weights to be in shape (batch_size x n_quantiles-1)
    weights = (tau_block[:, :-1] < estimated_quantiles.reshape(-1, 1)) * 1. + (
                tau_block[:, :-1] >= estimated_quantiles.reshape(-1, 1)) * (
                          tau_block[:, :-1] - estimated_quantiles.reshape(-1, 1)) / (
                          1 - estimated_quantiles.reshape(-1, 1))

    # now compute censored loss using
    # weight* censored value, + (1-weight)* fictionally large value
    y_max = y_max  # just use a really high value, larger than any data point we'll see
    loss_cens = torch.sum((c_true > 0) *
                          (weights * (y_pred[:, :-1] - t_true) * (
                                      (1 - tau_block[:, :-1]) - 1. * (y_pred[:, :-1] < t_true)) +
                           (1 - weights) * (y_pred[:, :-1] - y_max) * (
                                       (1 - tau_block[:, :-1]) - 1. * (y_pred[:, :-1] < y_max)))
                          , dim=1)
    # could drop *(y_pred[:,:-1]<y_max) as this will always be true, but incl. for completeness
    loss_cens = torch.mean(loss_cens)

    return loss_obs + loss_cens


class CensoredPinballLoss(nn.Module):
    def __init__(self, quantiles, use_cross_loss: bool = False, reduction: str = "mean"):
        super(CensoredPinballLoss, self).__init__()
        self.quan_levels = quantiles.reshape([1, -1])
        self.reduction = reduction
        self._t_max = None
        self.IS_USE_CROSS_LOSS = use_cross_loss

    @property
    def t_max(self):
        return self._t_max

    @t_max.setter
    def t_max(self, value):
        print("Setting t_max to {} for CQRNN.".format(value))
        self._t_max = value

    def forward(self, y_pred, y_true):
        t_true, e_true = y_true[:, 0], y_true[:, 1]
        t_true = t_true.reshape(-1, 1)
        e_true = e_true.reshape(-1, 1)
        c_true = 1 - e_true
        self.quan_levels = self.quan_levels.to(y_true.device)

        # y_pred is shape (n_batch, n_quantiles)
        # y_true is shape (n_batch, 1)
        # cen_indicator is shape (n_batch, 1)
        # use detach to figure out where to block
        # first figure out closest quantile (do for all observations)
        y_pred_detach = y_pred.detach()
        # do we need detach()? yes I think so, otherwise loss is affected, though it's argmin so gradients prob don't flow anyway

        # should do this outside loss really and subselect here if needed
        qaun_level_block = self.quan_levels.repeat((c_true.shape[0], 1)).to(y_true.device)

        loss_obs = quantile_loss(y_pred, t_true, c_true, self.quan_levels)

        # add in crossing loss
        if self.IS_USE_CROSS_LOSS:
            loss_obs += crossing_loss(y_pred)

        # use argmin to get nearest quantile
        torch_abs = torch.abs(t_true - y_pred_detach[:, :])
        estimated_quantiles = torch.max(
            qaun_level_block[:, :] * (
                        torch_abs == torch.min(torch_abs, dim=1).values.view(torch_abs.shape[0], 1)), dim=1).values

        # compute weights, eq 11, portnoy 2003
        # want weights to be in shape (batch_size x n_quantiles-1)
        weights = (qaun_level_block[:, :] < estimated_quantiles.reshape(-1, 1)) * 1. + (
                qaun_level_block[:, :] >= estimated_quantiles.reshape(-1, 1)) * (
                          qaun_level_block[:, :] - estimated_quantiles.reshape(-1, 1)) / (
                          1 - estimated_quantiles.reshape(-1, 1))

        # now compute censored loss using
        # weight * censored value, + (1-weight) * fictionally large value
        loss_cens = torch.sum((c_true > 0) *
                              (weights * (y_pred[:, :] - t_true) * (
                                      (1 - qaun_level_block[:, :]) - 1. * (y_pred[:, :] < t_true)) +
                               (1 - weights) * (y_pred[:, :] - self.t_max) * (
                                       (1 - qaun_level_block[:, :]) - 1. * (y_pred[:, :] < self.t_max)))
                              , dim=1)
        loss_cens = torch.mean(loss_cens)

        return loss_obs + loss_cens


def compute_km_cal(average_survival_curve, t_grids, t_true, e_true):
    device = average_survival_curve.device
    t_range = max(t_grids) - min(t_grids)

    km_model = KaplanMeierFitter().fit(t_true.cpu(), e_true.cpu())
    km_curve = torch.tensor(km_model.survival_function_at_times(t_grids.cpu().numpy()).values).to(device)

    assert len(km_curve) == len(average_survival_curve), ("The length of the average survival curve and "
                                                          "the KM curve should be the same.")
    # sum over the joint time coordinates
    km_cal = (1 / t_range) * torch.sum(torch.abs(average_survival_curve - km_curve))

    return km_cal


def compute_x_cal(cdf, e_true, gamma, n_bins=10):
    device = cdf.device
    is_alive = 1 - e_true.detach().clone()
    is_alive[cdf > 1. - 1e-8] = 0

    cdf = cdf.view(-1, 1)
    # print(cdf[:200])
    bin_width = 1.0 / n_bins
    bin_indices = torch.arange(n_bins).view(1, -1).float().to(device)
    bin_a = bin_indices * bin_width #+ 0.02*torch.rand(size=bin_indices.shape)
    noise = 1e-6 / n_bins * torch.rand(size=bin_indices.shape).to(device)
    cum_noise = torch.cumsum(noise, dim=1)
    bin_width = torch.tensor([bin_width] * n_bins).to(device) + cum_noise
    bin_b = bin_a + bin_width

    bin_b_max = bin_b[:, -1]
    bin_b = bin_b/bin_b_max
    bin_a[:, 1:] = bin_b[:, :-1]
    bin_width = bin_b - bin_a

    # CENSORED POINTS
    cdf_cens = cdf[is_alive.long() == 1]
    upper_diff_for_soft_cens = bin_b - cdf_cens
    # To solve optimization issue, we change the first left bin boundary to be -1.;
    # we change the last right bin boundary to be 2.
    bin_b[:, -1] = 2.
    bin_a[:, 0] = -1.
    lower_diff_cens = cdf_cens - bin_a # p - a
    upper_diff_cens = bin_b - cdf_cens # b - p
    diff_product_cens = lower_diff_cens * upper_diff_cens
    # NON-CENSORED POINTS

    # sigmoid(gamma*(p-a)*(b-p))
    bin_index_ohe = torch.sigmoid(gamma * diff_product_cens)
    exact_bins_next = torch.sigmoid(-gamma * lower_diff_cens)

    EPS = 1e-13
    right_censored_interval_size = 1 - cdf_cens + EPS

    # each point's distance from its bin's upper limit
    upper_diff_within_bin = (upper_diff_for_soft_cens * bin_index_ohe)

    # assigns weights to each full bin that is larger than the point
    # full_bin_assigned_weight = exact_bins*bin_width
    # 1 / right_censored_interval_size is the density of the uniform over [F(c),1]
    full_bin_assigned_weight = (exact_bins_next*bin_width.view(1,-1)/right_censored_interval_size.view(-1,1)).sum(0)
    partial_bin_assigned_weight = (upper_diff_within_bin/right_censored_interval_size).sum(0)
    assert full_bin_assigned_weight.shape == partial_bin_assigned_weight.shape, (full_bin_assigned_weight.shape, partial_bin_assigned_weight.shape)

    # NON-CENSORED POINTS
    cdf_uncens = cdf[is_alive.long() == 0]
    # compute p - a and b - p
    lower_diff = cdf_uncens - bin_a
    upper_diff = bin_b - cdf_uncens
    diff_product = lower_diff * upper_diff
    assert lower_diff.shape == upper_diff.shape, (lower_diff.shape, upper_diff.shape)
    assert lower_diff.shape == (cdf_uncens.shape[0], bin_a.shape[1])
    # NON-CENSORED POINTS

    # sigmoid(gamma*(p-a)*(b-p))
    soft_membership = torch.sigmoid(gamma*diff_product)
    fraction_in_bins = soft_membership.sum(0)
    # print('soft_membership', soft_membership)

    assert fraction_in_bins.shape == (n_bins, ), fraction_in_bins.shape

    frac_in_bins = (fraction_in_bins + full_bin_assigned_weight + partial_bin_assigned_weight) / cdf.shape[0]
    return torch.pow(frac_in_bins - bin_width, 2).sum()


class LikelihoodLogNormal(nn.Module):
    """Computes the negative log-likelihood of a batch of model predictions."""

    def __init__(self, reduction="mean", lam: float = 0.0, gamma: float = 10000, type: str = "x-cal"):
        super(LikelihoodLogNormal, self).__init__()
        assert reduction in ["mean", "sum"], "reduction must be one of 'mean', 'sum'"
        self.reduction = reduction
        self.lam = lam
        self.gamma = gamma
        self.type = type

    def forward(self, logits, y_true):
        t_true, e_true = y_true[:, 0], y_true[:, 1]
        mu = logits[:, 0]
        pre_log_sigma = logits[:, 1]
        log_sigma = F.softplus(pre_log_sigma) - 0.5
        sigma = log_sigma.clamp(max=10).exp()
        sigma = sigma + 1e-8

        dist = torch.distributions.LogNormal(mu, sigma)
        cdf = dist.cdf(t_true)
        survival = 1.0 - cdf
        t_grid = torch.unique(t_true[e_true == 1], sorted=True)
        average_survival = torch.empty_like(t_grid)
        for i in range(len(t_grid)):
            average_survival[i] = 1.0 - dist.cdf(t_grid[i]).mean()
        # cdf_all_points = dist.cdf(t_grid)
        # survival_all_points = 1.0 - cdf_all_points
        # average_survival = survival_all_points.mean()
        log_pdf = dist.log_prob(t_true)
        log_survival = safe_log(survival)

        loglikelihood = (1 - e_true) * log_survival + e_true * log_pdf

        nll = -1.0 * loglikelihood
        if self.reduction == "mean":
            nll = nll.mean(dim=-1)
        elif self.reduction == "sum":
            nll = nll.sum(dim=-1)

        if self.lam > 0:
            if self.type == "x-cal":
                cal_loss = compute_x_cal(cdf, e_true, self.gamma, n_bins=10)
            elif self.type == "sfm":
                cal_loss = compute_km_cal(average_survival, t_grid, t_true, e_true)
            else:
                raise ValueError("Invalid type for calibration loss.")
        else:
            cal_loss = 0
        return nll + self.lam * cal_loss


class CRPS(nn.Module):
    def __init__(self):
        super(CRPS, self).__init__()
        self.K = 32

    def I_ln(self, mu, scale, y, g):
        # integral of CDF^2 of lognormal times g;
        # using math from appendix A https://arxiv.org/pdf/1806.08324.pdf

        # X ~ N(mu, sigma) === > exp(X) ~ LogNormal(mu, sigma);
        # therefore, the Normal distribution we parameterize to compute the CDF uses the same sigma as scale
        norm = torch.distributions.normal.Normal(mu, scale) # CHECKED

        # approximation is as follows
        # let phi( x ) be the cdf of normal evaluated at x.
        # sum_k  0.5  * [ phi^2( log z_k) g(z_k) +  phi^2(log z_k-1) g(z_k-1) ] * [ z_k - z_k-1 ]

        # grid points to approximate the integral
        # creates K evenly spaced points from 1e-4 to 1
        # grid_points = torch.tensor(np.linspace(1e-4, 1, self.K).astype(np.float32)).to(mu.device)
        grid_points = torch.linspace(1e-4, 1, self.K).to(mu.device)

        # compute z_k-1 and \phi^2( log z_k-1) for k = 1; so z_0 and phi(z_0)
        z_km1 = y*grid_points[0]
        phi_km1 = norm.cdf(z_km1.log()).view(-1)
        summand_km1 =  phi_km1.pow(2)*g(z_km1).view(-1)

        # return value
        retval = 0.0

        # loop over k from 1 to K-1, both included
        for k in range(1, self.K):
            z_k = y*grid_points[k]

            # compute phi^2(log z_k)
            phi_k = norm.cdf(z_k.log()).view(-1)
            summand_k =  phi_k.pow(2)*g(z_k).view(-1)

            # accumulate the summand 0.5 [ phi^2( log z_k) g(z_k) +  phi^2(log z_km1) g(z_km1) ] * [ z_k - z_km1 ]
            retval = retval + 0.5*(summand_k + summand_km1)*(z_k - z_km1)

            # update z_k-1 and phi^2(z_k-1)
            z_km1 = z_k
            summand_km1 = summand_k

        return retval

    def CRPS_surv_ln(self, mu, scale_lognormal, time, censor):
        # argument sigma = s.exp is the scale of the logNormal distribution
        Y = time
        I = lambda y: self.I_ln(mu, scale_lognormal, y, lambda y_: y_*0 + 1)
        I_ = lambda y: self.I_ln(-mu, scale_lognormal, 1/(y + 1e-4), lambda y_: (y_+1e-4).pow(-1))

        crps = I(Y) + (1 - censor) * I_(Y)
        return crps

    def forward(self, logits, y_true):
        t_true, e_true = y_true[:, 0], y_true[:, 1]
        is_alive = 1 - e_true
        mu = logits[:, 0]
        pre_log_sigma = logits[:, 1]
        log_sigma = F.softplus(pre_log_sigma) - 0.5
        sigma = log_sigma.clamp(max=10).exp()
        sigma = sigma + 1e-8

        scale_lognormal = sigma
        # what we use for CDF for dcal pred = torch.distributions.LogNormal(mu, scale_lognormal)
        loss = self.CRPS_surv_ln(mu, scale_lognormal, t_true, is_alive)

        loss = loss.mean()
        return loss
