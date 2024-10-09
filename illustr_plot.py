#%%
# prerequisites
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from scipy.stats import gaussian_kde
from scipy.integrate import cumtrapz



color_list = sns.color_palette("Set2", 10)
color_group = sns.color_palette("deep", 10)

custom_params = {"axes.spines.right": False, "axes.spines.top": False,}
plt.rcParams['figure.constrained_layout.use'] = True
sns.set_theme(style="white", rc=custom_params)

plt.rc('axes', titlesize=10)     # fontsize of the axes title
plt.rc('axes', labelsize=10)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
plt.rc('legend', fontsize=8)    # legend fontsize
plt.rc('figure', titlesize=10)  # fontsize of the figure title
height_unit = 2
width_unit = 3.0
plt.rcParams['font.size'] = 12


def weibull_pdf(x, scale, shape):
    return (shape / scale) * (x / scale)**(shape - 1) * np.exp(-(x / scale)**shape)


def gamma_pdf(x, scale, shape):
    return (x ** (shape - 1) * np.exp(-x / scale)) / (math.gamma(shape) * scale ** shape)


def lognormal_pdf(x, mu, sigma):
    return (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(-(np.log(x) - mu) ** 2 / (2 * sigma ** 2))


def loglogistic_pdf(x, scale, shape):
    return (shape / scale) * (x / scale) ** (shape - 1) / (1 + (x / scale) ** shape) ** 2

#%%
# Example 1, individual calibration
end_time = 3.5

# set seed for reproducibility
np.random.seed(12345)

N = 100000
s1_1 = 0.5 * np.random.weibull(a=1, size=int(0.3*N))   # generate your data sample with N elements
s1_2 = 2.8 * np.random.weibull(a=7, size=int(0.7*N))
s1 = np.append(s1_1, s1_2)
kde = gaussian_kde(s1)


# sample same data points from the distribution
sample_size = 15
# sample = kde.resample(size=sample_size).reshape(-1)
# # ps = kde(sample)
# # cdfs = cumtrapz(ps, sample, initial=0)
# death_ids = (sample / 0.001).astype(int)

dist_space = np.arange(0, end_time, 0.001)
p_1 = kde(dist_space)
cdf_1 = cumtrapz(p_1, dist_space, initial=0)

sample_cdf = np.random.uniform(size=sample_size)
# find the corresponding x values
sample_idx = []
for cdf_i in sample_cdf:
    sample_idx.append(np.where(cdf_1 >= cdf_i)[0][0])


fig1, axes1 = plt.subplots(nrows=1, ncols=1, figsize=(width_unit, height_unit))

axes1.plot(dist_space, 1 - cdf_1, label='Patient A', c=color_group[0], linewidth=2, ls='solid', clip_on=False, zorder=10)
# axes1.scatter(sample, 1 - cdf_1[death_ids], marker='*', c=color_group[0], s=100)
axes1.scatter(dist_space[sample_idx], 1 - sample_cdf, marker='*', c=color_group[0], s=100, clip_on=False, zorder=10)
axes1.set_ylim([-0.0, 1.0])
axes1.set_xlim([0, end_time])
axes1.set_yticks([0, 1/3, 2/3, 1], [r'$0$',r'$\frac{1}{3}$', r'$\frac{2}{3}$', r'$1$'])
axes1.grid(axis='y')
axes1.set_xlabel('Days')
axes1.set_ylabel('Survival Prob.')

plt.tight_layout()
plt.savefig(f'figs/illustration/ind_cal.png', dpi=400, transparent=True)

# probs_at_event = 1 - cdf_1[death_ids]
probs_at_event = 1 - sample_cdf
hist = np.histogram(probs_at_event, bins=[0, 1/3, 2/3, 1])
probs = hist[0] / sample_size
cum_probs = np.cumsum(probs)
cum_probs = np.insert(cum_probs, 0, 0)
fig2, axes2 = plt.subplots(nrows=1, ncols=2, figsize=(width_unit*0.6*2, height_unit), width_ratios=[1, 1.4])
axes2[0].hist(probs_at_event, bins=[0, 1/3, 2/3, 1], color=color_group[0], rwidth=1, orientation='horizontal')
axes2[0].set_ylim([-0.0, 1.0])
axes2[0].axvline(sample_size/3, ls='dashed', c='grey', label='Perfect Cal.')
axes2[0].set_xlabel('Counts')
axes2[0].set_ylabel('Survival Prob.')
axes2[0].grid(axis='y')
axes2[0].set_xlim([0, 7.5])
axes2[0].set_xticks([0, 5])
axes2[0].set_yticks([0, 1/3, 2/3, 1], [r'$0$',r'$\frac{1}{3}$', r'$\frac{2}{3}$', r'$1$'])
# axes2[0].legend()

axes2[1].plot([0, 1], [0, 1], ls='dashed', c='grey', label='Perfect Cal.')
axes2[1].plot([0, 1/3, 2/3, 1], cum_probs, marker='o', markersize=10, c=color_group[0], linewidth=2,
              clip_on=False, zorder=10)
axes2[1].set_ylim([-0.0, 1.0])
axes2[1].set_xlim([-0.0, 1.0])
axes2[1].set_xticks([0, 1/3, 2/3, 1], [r'$0$', '1/3', '2/3', '1'])
axes2[1].set_yticks([0, 1/3, 2/3, 1], [r'$0$',r'$\frac{1}{3}$', r'$\frac{2}{3}$', r'$1$'])
axes2[1].set_xlabel('Predicted Prob.')
axes2[1].set_ylabel('Observed Prob.')
axes2[1].grid(False)
axes2[1].legend(loc='upper left')

plt.tight_layout()
plt.savefig(f'figs/illustration/ind_cal_hist.png', dpi=400, transparent=True)


#%%
# Example 2, conditional calibration
e_1 = 2.6
e_2 = 1.5
e_3 = 0.6
e_4 = 2.1
e_5 = 1.8
e_6 = 1.2
end_time = 3
N = 100000
s1_1 = 0.5 * np.random.weibull(a=1, size=int(0.3*N))   # generate your data sample with N elements
s1_2 = 2.8 * np.random.weibull(a=7, size=int(0.7*N))
s1 = np.append(s1_1, s1_2)
kde = gaussian_kde(s1)
# these are the values over which your kernel will be evaluated
dist_space = np.arange(0, end_time, 0.001)
p_1 = kde(dist_space)
# plt.plot( dist_space, pdf )
cdf_1 = cumtrapz(p_1, dist_space, initial=0)
death_idx_1 = int(e_1/0.001)
# ab_1 = AnnotationBbox(death_imagebox, (e_1, 1 - cdf_1[death_idx_1]), frameon = False)

p_2 = weibull_pdf(dist_space, 0.8, 1.5)
cdf_2 = cumtrapz(p_2, dist_space, initial=0)
death_idx_2 = int(e_2/0.001)
p_3 = gamma_pdf(dist_space, 0.5, 3)
cdf_3 = cumtrapz(p_3, dist_space, initial=0)
death_idx_3 = int(e_3/0.001)

s4_1 = 0.5 * np.random.gamma(shape=0.7, scale=0.8, size=int(0.4*N))   # generate your data sample with N elements
s4_2 = 2.1 * np.random.gamma(shape=1.4, scale=1.5, size=int(0.6*N))
s4 = np.append(s4_1, s4_2)
kde = gaussian_kde(s4)
p_4 = kde(dist_space)
cdf_4 = cumtrapz(p_4, dist_space, initial=0)
death_idx_4 = int(e_4/0.001)

p_5 = lognormal_pdf(dist_space, 0.7, 0.15)
p_5[0] = 0
cdf_5 = cumtrapz(p_5, dist_space, initial=0)
death_idx_5 = int(e_5/0.001)

p_6 = loglogistic_pdf(dist_space, 0.5, 1)
cdf_6 = cumtrapz(p_6, dist_space, initial=0)
death_idx_6 = int(e_6/0.001)


fig1, axes1 = plt.subplots(nrows=1, ncols=1, figsize=(width_unit, height_unit))

# d-calibration
axes1.plot(dist_space, 1 - cdf_1, label='Patient A', c=color_group[1], linewidth=2, clip_on=False, zorder=10)
axes1.plot(dist_space, 1 - cdf_2, label='Patient B', c=color_group[1], linewidth=2, clip_on=False, zorder=10)
axes1.plot(dist_space, 1 - cdf_3, label='Patient C', c=color_group[1], linewidth=2, clip_on=False, zorder=10)
axes1.plot(dist_space, 1 - cdf_4, label='Patient D', c=color_group[0], linewidth=2, clip_on=False, zorder=10)
axes1.plot(dist_space, 1 - cdf_5, label='Patient E', c=color_group[0], linewidth=2, clip_on=False, zorder=10)
axes1.plot(dist_space, 1 - cdf_6, label='Patient F', c=color_group[0], linewidth=2, clip_on=False, zorder=10)
axes1.scatter(e_1, 1 - cdf_1[death_idx_1], marker='*', c=color_list[0], s=100, clip_on=False, zorder=20)
axes1.scatter(e_2, 1 - cdf_2[death_idx_2], marker='*', c=color_list[1], s=100, clip_on=False, zorder=20)
axes1.scatter(e_3, 1 - cdf_3[death_idx_3], marker='*', c=color_list[2], s=100, clip_on=False, zorder=20)
axes1.scatter(e_4, 1 - cdf_4[death_idx_4], marker='*', c=color_list[3], s=100, clip_on=False, zorder=20)
axes1.scatter(e_5, 1 - cdf_5[death_idx_5], marker='*', c=color_list[4], s=100, clip_on=False, zorder=20)
axes1.scatter(e_6, 1 - cdf_6[death_idx_6], marker='*', c=color_list[5], s=100, clip_on=False, zorder=20)

# axes[1, 2].spines.set_color('black')
axes1.set_ylim([-0.0, 1.0])
axes1.set_xlim([0, end_time])
axes1.set_yticks([0, 1/3, 2/3, 1], [r'$0$',r'$\frac{1}{3}$', r'$\frac{2}{3}$', r'$1$'])
axes1.grid(axis='y')
axes1.set_xlabel('Days')
axes1.set_ylabel('Survival Prob.')


plt.tight_layout()
plt.savefig(f'figs/illustration/cond_cal.png', dpi=400, transparent=True)

probs_at_event_g1 = [1 - cdf_1[death_idx_1], 1 - cdf_2[death_idx_2], 1 - cdf_3[death_idx_3]]
probs_at_event_g2 = [1 - cdf_4[death_idx_4], 1 - cdf_5[death_idx_5], 1 - cdf_6[death_idx_6]]

hist_g1 = np.histogram(probs_at_event_g1, bins=[0, 1/3, 2/3, 1])
probs_g1 = hist_g1[0] / 3
cum_probs_g1 = np.cumsum(probs_g1)

hist_g2 = np.histogram(probs_at_event_g2, bins=[0, 1/3, 2/3, 1])
probs_g2 = hist_g2[0] / 3
cum_probs_g2 = np.cumsum(probs_g2)

cum_probs_g1 = np.insert(cum_probs_g1, 0, 0)
cum_probs_g2 = np.insert(cum_probs_g2, 0, 0)
fig2, axes2 = plt.subplots(nrows=1, ncols=2, figsize=(width_unit*0.6*2, height_unit), width_ratios=[1, 1.4])
axes2[0].hist([probs_at_event_g1, probs_at_event_g2], bins=[0, 1/3, 2/3, 1],
              color=[color_group[1], color_group[0]],
              rwidth=1, orientation='horizontal', stacked=True, )
axes2[0].set_ylim([-0.0, 1.0])
axes2[0].axvline(2, ls='dashed', c='grey', label='Perfect Cal.')
axes2[0].set_xlabel('Counts')
axes2[0].set_ylabel('Survival Prob.')
axes2[0].grid(axis='y')
axes2[0].set_xlim([0, 3])
axes2[0].set_xticks([0, 2])
axes2[0].set_yticks([0, 1/3, 2/3, 1], [r'$0$',r'$\frac{1}{3}$', r'$\frac{2}{3}$', r'$1$'])
# axes2[0].legend()

# axes2[1].plot([0, 1], [0, 1], ls='dashed', c='grey', label='Perfect Cal.')
axes2[1].plot([0, 1/3, 2/3, 1], cum_probs_g1, marker='o', markersize=10, c=color_group[1], linewidth=2,
              clip_on=False, zorder=10, label=r'$x=0$')
axes2[1].plot([0, 1/3, 2/3, 1], cum_probs_g2, marker='o', markersize=10, c=color_group[0], linewidth=2,
              clip_on=False, zorder=10, label=r'$x=1$')
axes2[1].plot([0, 1/3, 2/3, 1], 0.5 * (cum_probs_g1 + cum_probs_g2), ls='dashed', marker='o', markersize=10,
              c='grey', linewidth=2, clip_on=False, zorder=10, label='Marginal')
axes2[1].set_ylim([-0.0, 1.0])
axes2[1].set_xlim([-0.0, 1.0])
axes2[1].set_xticks([0, 1/3, 2/3, 1], [r'$0$', '1/3', '2/3', '1'])
axes2[1].set_yticks([0, 1/3, 2/3, 1], [r'$0$',r'$\frac{1}{3}$', r'$\frac{2}{3}$', r'$1$'])
axes2[1].set_xlabel('Predicted Prob.')
axes2[1].set_ylabel('Observed Prob.')
axes2[1].grid(False)
axes2[1].legend(framealpha=0.5, loc='upper left').set_zorder(20)

plt.tight_layout()
plt.savefig(f'figs/illustration/cond_cal_hist.png', dpi=400, transparent=True)



#%%
# Example 3, marginal calibration
e_1 = 2.6
e_2 = 1.5
e_3 = 0.6
e_4 = 2.1
e_5 = 1.8
e_6 = 1.2
end_time = 3
N = 100000
s1_1 = 0.5 * np.random.weibull(a=1, size=int(0.35*N))   # generate your data sample with N elements
s1_2 = 2.6 * np.random.weibull(a=7, size=int(0.65*N))
s1 = np.append(s1_1, s1_2)
kde = gaussian_kde(s1)
# these are the values over which your kernel will be evaluated
dist_space = np.arange(0, end_time, 0.001)
p_1 = kde(dist_space)
# plt.plot( dist_space, pdf )
cdf_1 = cumtrapz(p_1, dist_space, initial=0)
death_idx_1 = int(e_1/0.001)
# ab_1 = AnnotationBbox(death_imagebox, (e_1, 1 - cdf_1[death_idx_1]), frameon = False)

p_2 = weibull_pdf(dist_space, 0.6, 1.8)
cdf_2 = cumtrapz(p_2, dist_space, initial=0)
death_idx_2 = int(e_2/0.001)

p_3 = loglogistic_pdf(dist_space, 0.6, 2)
cdf_3 = cumtrapz(p_3, dist_space, initial=0)
death_idx_3 = int(e_3/0.001)


s4_1 = 0.5 * np.random.gamma(shape=0.7, scale=0.8, size=int(0.15*N))   # generate your data sample with N elements
s4_2 = 3.0 * np.random.gamma(shape=1.4, scale=1.5, size=int(0.85*N))
s4 = np.append(s4_1, s4_2)
kde = gaussian_kde(s4)
p_4 = kde(dist_space)
cdf_4 = cumtrapz(p_4, dist_space, initial=0)
death_idx_4 = int(e_4/0.001)

p_5 = lognormal_pdf(dist_space, 0.7, 0.2)
p_5[0] = 0
cdf_5 = cumtrapz(p_5, dist_space, initial=0)
death_idx_5 = int(e_5/0.001)

p_6 = gamma_pdf(dist_space, 0.4, 3)
cdf_6 = cumtrapz(p_6, dist_space, initial=0)
death_idx_6 = int(e_6/0.001)


fig1, axes1 = plt.subplots(nrows=1, ncols=1, figsize=(width_unit, height_unit))

# d-calibration
axes1.plot(dist_space, 1 - cdf_1, label='Patient A', c=color_group[1], linewidth=2, clip_on=False, zorder=10)
axes1.plot(dist_space, 1 - cdf_2, label='Patient B', c=color_group[1], linewidth=2, clip_on=False, zorder=10)
axes1.plot(dist_space, 1 - cdf_3, label='Patient C', c=color_group[1], linewidth=2, clip_on=False, zorder=10)
axes1.plot(dist_space, 1 - cdf_4, label='Patient D', c=color_group[0], linewidth=2, clip_on=False, zorder=10)
axes1.plot(dist_space, 1 - cdf_5, label='Patient E', c=color_group[0], linewidth=2, clip_on=False, zorder=10)
axes1.plot(dist_space, 1 - cdf_6, label='Patient F', c=color_group[0], linewidth=2, clip_on=False, zorder=10)
axes1.scatter(e_1, 1 - cdf_1[death_idx_1], marker='*', c=color_list[0], s=100, clip_on=False, zorder=20)
axes1.scatter(e_2, 1 - cdf_2[death_idx_2], marker='*', c=color_list[1], s=100, clip_on=False, zorder=20)
axes1.scatter(e_3, 1 - cdf_3[death_idx_3], marker='*', c=color_list[2], s=100, clip_on=False, zorder=20)
axes1.scatter(e_4, 1 - cdf_4[death_idx_4], marker='*', c=color_list[3], s=100, clip_on=False, zorder=20)
axes1.scatter(e_5, 1 - cdf_5[death_idx_5], marker='*', c=color_list[4], s=100, clip_on=False, zorder=20)
axes1.scatter(e_6, 1 - cdf_6[death_idx_6], marker='*', c=color_list[5], s=100, clip_on=False, zorder=20)

# axes[1, 2].spines.set_color('black')
axes1.set_ylim([-0.0, 1.0])
axes1.set_xlim([0, end_time])
axes1.set_yticks([0, 1/3, 2/3, 1], [r'$0$',r'$\frac{1}{3}$', r'$\frac{2}{3}$', r'$1$'])
axes1.grid(axis='y')
axes1.set_xlabel('Days')
axes1.set_ylabel('Survival Prob.')


plt.tight_layout()
plt.savefig(f'figs/illustration/marg_cal.png', dpi=400, transparent=True)

probs_at_event_g1 = [1 - cdf_1[death_idx_1], 1 - cdf_2[death_idx_2], 1 - cdf_3[death_idx_3]]
probs_at_event_g2 = [1 - cdf_4[death_idx_4], 1 - cdf_5[death_idx_5], 1 - cdf_6[death_idx_6]]

hist_g1 = np.histogram(probs_at_event_g1, bins=[0, 1/3, 2/3, 1])
probs_g1 = hist_g1[0] / 3
cum_probs_g1 = np.cumsum(probs_g1)

hist_g2 = np.histogram(probs_at_event_g2, bins=[0, 1/3, 2/3, 1])
probs_g2 = hist_g2[0] / 3
cum_probs_g2 = np.cumsum(probs_g2)

cum_probs_g1 = np.insert(cum_probs_g1, 0, 0)
cum_probs_g2 = np.insert(cum_probs_g2, 0, 0)
fig2, axes2 = plt.subplots(nrows=1, ncols=2, figsize=(width_unit*0.6*2, height_unit), width_ratios=[1, 1.4])
axes2[0].hist([probs_at_event_g1, probs_at_event_g2], bins=[0, 1/3, 2/3, 1],
              color=[color_group[1], color_group[0]],
              rwidth=1, orientation='horizontal', stacked=True, )
axes2[0].set_ylim([-0.0, 1.0])
axes2[0].axvline(2, ls='dashed', c='grey', label='Perfect Cal.')
axes2[0].set_xlabel('Counts')
axes2[0].set_ylabel('Survival Prob.')
axes2[0].grid(axis='y')
axes2[0].set_xlim([0, 3])
axes2[0].set_xticks([0, 2])
axes2[0].set_yticks([0, 1/3, 2/3, 1], [r'$0$',r'$\frac{1}{3}$', r'$\frac{2}{3}$', r'$1$'])
# axes2[0].legend()

# axes2[1].plot([0, 1], [0, 1], ls='dashed', c='grey', label='Perfect Cal.')
axes2[1].plot([0, 1/3, 2/3, 1], cum_probs_g1, marker='o', markersize=10, c=color_group[1], linewidth=2,
              clip_on=False, zorder=10, label=r'$x=0$')
axes2[1].plot([0, 1/3, 2/3, 1], cum_probs_g2, marker='o', markersize=10, c=color_group[0], linewidth=2,
              clip_on=False, zorder=10, label=r'$x=1$')
axes2[1].plot([0, 1/3, 2/3, 1], 0.5 * (cum_probs_g1 + cum_probs_g2), ls='dashed', marker='o', markersize=10,
              c='grey', linewidth=2, clip_on=False, zorder=10, label='Marginal')
axes2[1].set_ylim([-0.0, 1.0])
axes2[1].set_xlim([-0.0, 1.0])
axes2[1].set_xticks([0, 1/3, 2/3, 1], [r'$0$', '1/3', '2/3', '1'])
axes2[1].set_yticks([0, 1/3, 2/3, 1], [r'$0$',r'$\frac{1}{3}$', r'$\frac{2}{3}$', r'$1$'])
axes2[1].set_xlabel('Predicted Prob.')
axes2[1].set_ylabel('Observed Prob.')
axes2[1].grid(False)
axes2[1].legend(framealpha=0.5).set_zorder(20)

plt.tight_layout()
plt.savefig(f'figs/illustration/marg_cal_hist.png', dpi=400, transparent=True)



#%%
# Example 4, non calibration
e_1 = 2.6
e_2 = 1.5
e_3 = 0.6
e_4 = 2.1
e_5 = 1.8
e_6 = 1.2
end_time = 3
N = 100000
s1_1 = 0.5 * np.random.weibull(a=1, size=int(0.3*N))   # generate your data sample with N elements
s1_2 = 2.8 * np.random.weibull(a=7, size=int(0.7*N))
s1 = np.append(s1_1, s1_2)
kde = gaussian_kde(s1)
# these are the values over which your kernel will be evaluated
dist_space = np.arange(0, end_time, 0.001)
p_1 = kde(dist_space)
# plt.plot( dist_space, pdf )
cdf_1 = cumtrapz(p_1, dist_space, initial=0)
death_idx_1 = int(e_1/0.001)
# ab_1 = AnnotationBbox(death_imagebox, (e_1, 1 - cdf_1[death_idx_1]), frameon = False)

p_2 = weibull_pdf(dist_space, 0.4, 1.7)
cdf_2 = cumtrapz(p_2, dist_space, initial=0)
death_idx_2 = int(e_2/0.001)
p_3 = gamma_pdf(dist_space, 0.5, 3)
cdf_3 = cumtrapz(p_3, dist_space, initial=0)
death_idx_3 = int(e_3/0.001)

s4_1 = 0.5 * np.random.gamma(shape=0.7, scale=0.8, size=int(0.4*N))   # generate your data sample with N elements
s4_2 = 1.2 * np.random.gamma(shape=1.2, scale=1.4, size=int(0.6*N))
s4 = np.append(s4_1, s4_2)
kde = gaussian_kde(s4)
p_4 = kde(dist_space)
cdf_4 = cumtrapz(p_4, dist_space, initial=0)
death_idx_4 = int(e_4/0.001)

p_5 = lognormal_pdf(dist_space, 0.62, 0.15)
p_5[0] = 0
cdf_5 = cumtrapz(p_5, dist_space, initial=0)
death_idx_5 = int(e_5/0.001)

p_6 = loglogistic_pdf(dist_space, 0.5, 1.5)
cdf_6 = cumtrapz(p_6, dist_space, initial=0)
death_idx_6 = int(e_6/0.001)


fig1, axes1 = plt.subplots(nrows=1, ncols=1, figsize=(width_unit*0.75*1.2, height_unit*1.2))

# d-calibration
axes1.plot(dist_space, 1 - cdf_1, label='Patient A', c=color_group[1], linewidth=2, clip_on=False, zorder=10)
axes1.plot(dist_space, 1 - cdf_2, label='Patient B', c=color_group[1], linewidth=2, clip_on=False, zorder=10)
axes1.plot(dist_space, 1 - cdf_3, label='Patient C', c=color_group[1], linewidth=2, clip_on=False, zorder=10)
axes1.plot(dist_space, 1 - cdf_4, label='Patient D', c=color_group[0], linewidth=2, clip_on=False, zorder=10)
axes1.plot(dist_space, 1 - cdf_5, label='Patient E', c=color_group[0], linewidth=2, clip_on=False, zorder=10)
axes1.plot(dist_space, 1 - cdf_6, label='Patient F', c=color_group[0], linewidth=2, clip_on=False, zorder=10)
axes1.scatter(e_1, 1 - cdf_1[death_idx_1], marker='*', c=color_list[0], s=100, clip_on=False, zorder=20)
axes1.scatter(e_2, 1 - cdf_2[death_idx_2], marker='*', c=color_list[1], s=100, clip_on=False, zorder=20)
axes1.scatter(e_3, 1 - cdf_3[death_idx_3], marker='*', c=color_list[2], s=100, clip_on=False, zorder=20)
axes1.scatter(e_4, 1 - cdf_4[death_idx_4], marker='*', c=color_list[3], s=100, clip_on=False, zorder=20)
axes1.scatter(e_5, 1 - cdf_5[death_idx_5], marker='*', c=color_list[4], s=100, clip_on=False, zorder=20)
axes1.scatter(e_6, 1 - cdf_6[death_idx_6], marker='*', c=color_list[5], s=100, clip_on=False, zorder=20)

# axes[1, 2].spines.set_color('black')
axes1.set_ylim([-0.0, 1.0])
axes1.set_xlim([0, end_time])
axes1.set_yticks([0, 1/3, 2/3, 1], [r'$0$',r'$\frac{1}{3}$', r'$\frac{2}{3}$', r'$1$'])
axes1.grid(axis='y')
axes1.set_xlabel('Days')
axes1.set_ylabel('Survival Prob.')


plt.tight_layout()
plt.savefig(f'figs/illustration/non_cal.png', dpi=400, transparent=True)

probs_at_event_g1 = [1 - cdf_1[death_idx_1], 1 - cdf_2[death_idx_2], 1 - cdf_3[death_idx_3]]
probs_at_event_g2 = [1 - cdf_4[death_idx_4], 1 - cdf_5[death_idx_5], 1 - cdf_6[death_idx_6]]

hist_g1 = np.histogram(probs_at_event_g1, bins=[0, 1/3, 2/3, 1])
probs_g1 = hist_g1[0] / 3
cum_probs_g1 = np.cumsum(probs_g1)

hist_g2 = np.histogram(probs_at_event_g2, bins=[0, 1/3, 2/3, 1])
probs_g2 = hist_g2[0] / 3
cum_probs_g2 = np.cumsum(probs_g2)

cum_probs_g1 = np.insert(cum_probs_g1, 0, 0)
cum_probs_g2 = np.insert(cum_probs_g2, 0, 0)
fig2, axes2 = plt.subplots(nrows=1, ncols=2, figsize=(width_unit*0.5*1.2*2, height_unit*1.2), width_ratios=[1, 1.4])
axes2[0].hist([probs_at_event_g1, probs_at_event_g2], bins=[0, 1/3, 2/3, 1],
              color=[color_group[1], color_group[0]],
              rwidth=1, orientation='horizontal', stacked=True, )
axes2[0].set_ylim([-0.0, 1.0])
axes2[0].axvline(2, ls='dashed', c='grey', label='Perfect Cal.')
axes2[0].set_xlabel('Counts')
axes2[0].set_ylabel('Survival Prob.')
axes2[0].grid(axis='y')
axes2[0].set_xlim([0, 3.5])
axes2[0].set_xticks([0, 2])
axes2[0].set_yticks([0, 1/3, 2/3, 1], [r'$0$',r'$\frac{1}{3}$', r'$\frac{2}{3}$', r'$1$'])
# axes2[0].legend()

# axes2[1].plot([0, 1], [0, 1], ls='dashed', c='grey', label='Perfect Cal.')
axes2[1].plot([0, 1/3, 2/3, 1], cum_probs_g1, marker='o', markersize=10, c=color_group[1], linewidth=2,
              clip_on=False, zorder=10, label=r'$x=0$')
axes2[1].plot([0, 1/3, 2/3, 1], cum_probs_g2, marker='o', markersize=10, c=color_group[0], linewidth=2,
              clip_on=False, zorder=10, label=r'$x=1$')
axes2[1].plot([0, 1/3, 2/3, 1], 0.5 * (cum_probs_g1 + cum_probs_g2), ls='dashed', marker='o', markersize=10,
              c='grey', linewidth=2, clip_on=False, zorder=10, label='Marginal')
axes2[1].set_ylim([-0.0, 1.0])
axes2[1].set_xlim([-0.0, 1.0])
axes2[1].set_xticks([0, 1/3, 2/3, 1], [r'$0$', '1/3', '2/3', '1'])
axes2[1].set_yticks([0, 1/3, 2/3, 1], [r'$0$',r'$\frac{1}{3}$', r'$\frac{2}{3}$', r'$1$'])
axes2[1].set_xlabel('Predicted Prob.')
axes2[1].set_ylabel('Observed Prob.')
axes2[1].grid(False)
axes2[1].legend(framealpha=0.5).set_zorder(20)

plt.tight_layout()
plt.savefig(f'figs/illustration/non_cal_hist.png', dpi=400, transparent=True)

#%%

# iPET method

probs_at_event = probs_at_event_g1 + probs_at_event_g2
q = np.quantile(probs_at_event, [1/3, 2/3, ])

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(width_unit*0.75*1.2, height_unit*1.2))

last_time = end_time + 0.3
disc_1 = [last_time] + [dist_space[np.where(cdf_1 >q_)[0][0]] for q_ in 1 - q] + [0]
disc_2 = [last_time] + [dist_space[np.where(cdf_2 >q_)[0][0]] for q_ in 1 - q] + [0]
disc_3 = [last_time] + [dist_space[np.where(cdf_3 >q_)[0][0]] for q_ in 1 - q] + [0]
disc_4 = [last_time] + [dist_space[np.where(cdf_4 >q_)[0][0]] for q_ in 1 - q] + [0]
disc_5 = [last_time] + [dist_space[np.where(cdf_5 >q_)[0][0]] for q_ in 1 - q] + [0]
disc_6 = [last_time] + [dist_space[np.where(cdf_6 >q_)[0][0]] for q_ in 1 - q] + [0]

pet_1 = np.interp(e_1, disc_1[::-1], [1, 2/3, 1/3, 0])
pet_2 = np.interp(e_2, disc_2[::-1], [1, 2/3, 1/3, 0])
pet_3 = np.interp(e_3, disc_3[::-1], [1, 2/3, 1/3, 0])
pet_4 = np.interp(e_4, disc_4[::-1], [1, 2/3, 1/3, 0])
pet_5 = np.interp(e_5, disc_5[::-1], [1, 2/3, 1/3, 0])
pet_6 = np.interp(e_6, disc_6[::-1], [1, 2/3, 1/3, 0])

# d-calibration
axes.plot(dist_space, 1 - cdf_1, label='Patient A', c=color_group[1], linewidth=2, clip_on=False, zorder=10)
axes.plot(dist_space, 1 - cdf_2, label='Patient B', c=color_group[1], linewidth=2, clip_on=False, zorder=10)
axes.plot(dist_space, 1 - cdf_3, label='Patient C', c=color_group[1], linewidth=2, clip_on=False, zorder=10)
axes.plot(dist_space, 1 - cdf_4, label='Patient D', c=color_group[0], linewidth=2, clip_on=False, zorder=10)
axes.plot(dist_space, 1 - cdf_5, label='Patient E', c=color_group[0], linewidth=2, clip_on=False, zorder=10)
axes.plot(dist_space, 1 - cdf_6, label='Patient F', c=color_group[0], linewidth=2, clip_on=False, zorder=10)
axes.scatter(e_1, 1 - cdf_1[death_idx_1], marker='*', c=color_list[0], s=100, clip_on=False, zorder=20)
axes.scatter(e_2, 1 - cdf_2[death_idx_2], marker='*', c=color_list[1], s=100, clip_on=False, zorder=20)
axes.scatter(e_3, 1 - cdf_3[death_idx_3], marker='*', c=color_list[2], s=100, clip_on=False, zorder=20)
axes.scatter(e_4, 1 - cdf_4[death_idx_4], marker='*', c=color_list[3], s=100, clip_on=False, zorder=20)
axes.scatter(e_5, 1 - cdf_5[death_idx_5], marker='*', c=color_list[4], s=100, clip_on=False, zorder=20)
axes.scatter(e_6, 1 - cdf_6[death_idx_6], marker='*', c=color_list[5], s=100, clip_on=False, zorder=20)
axes.scatter(disc_1[1:-1], [q[0], q[1]], edgecolors=color_list[0], linestyle='--', facecolors='none', s=60)
axes.scatter(disc_2[1:-1], [q[0], q[1]], edgecolors=color_list[1], linestyle='--', facecolors='none', s=60)
axes.scatter(disc_3[1:-1], [q[0], q[1]], edgecolors=color_list[2], linestyle='--', facecolors='none', s=60)
axes.scatter(disc_4[1:-1], [q[0], q[1]], edgecolors=color_list[3], linestyle='--', facecolors='none', s=60)
axes.scatter(disc_5[1:-1], [q[0], q[1]], edgecolors=color_list[4], linestyle='--', facecolors='none', s=60)
axes.scatter(disc_6[1:-1], [q[0], q[1]], edgecolors=color_list[5], linestyle='--', facecolors='none', s=60)
axes.axhline(q[0], ls='-', c='dimgrey', label=r'$Q(\frac{1}{3})$', zorder=5)
axes.axhline(q[1], ls='-', c='dimgrey', label=r'$Q(\frac{2}{3})$', zorder=5)

axes.set_ylim([-0.0, 1.0])
axes.set_xlim([0, end_time])
axes.set_yticks([0, 1/3, 2/3, 1], [r'$0$',r'$\frac{1}{3}$', r'$\frac{2}{3}$', r'$1$'])
axes.grid(axis='y')
axes.set_xlabel('Days')
axes.set_ylabel('Survival Prob.')
plt.tight_layout()
plt.savefig(f'figs/illustration/iPET_step1.png', dpi=400, transparent=True)


fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(width_unit*0.75*1.2, height_unit*1.2))

axes.scatter(e_1, pet_1, marker='*', c=color_list[0], s=100, clip_on=False, zorder=20)
axes.scatter(e_2, pet_2, marker='*', c=color_list[1], s=100, clip_on=False, zorder=20)
axes.scatter(e_3, pet_3, marker='*', c=color_list[2], s=100, clip_on=False, zorder=20)
axes.scatter(e_4, pet_4, marker='*', c=color_list[3], s=100, clip_on=False, zorder=20)
axes.scatter(e_5, pet_5, marker='*', c=color_list[4], s=100, clip_on=False, zorder=20)
axes.scatter(e_6, pet_6, marker='*', c=color_list[5], s=100, clip_on=False, zorder=20)
axes.plot(disc_1, [0, 1/3, 2/3, 1], c=color_group[1], linestyle='-',  zorder=10)
axes.plot(disc_2, [0, 1/3, 2/3, 1], c=color_group[1], linestyle='-',  zorder=10)
axes.plot(disc_3, [0, 1/3, 2/3, 1], c=color_group[1], linestyle='-',  zorder=10)
axes.plot(disc_4, [0, 1/3, 2/3, 1], c=color_group[0], linestyle='-',  zorder=10)
axes.plot(disc_5, [0, 1/3, 2/3, 1], c=color_group[0], linestyle='-',  zorder=10)
axes.plot(disc_6, [0, 1/3, 2/3, 1], c=color_group[0], linestyle='-',  zorder=10)
axes.scatter(disc_1[1:-1], [1/3, 2/3], edgecolors=color_list[0], linestyle='--', facecolors='none', s=60)
axes.scatter(disc_2[1:-1], [1/3, 2/3], edgecolors=color_list[1], linestyle='--', facecolors='none', s=60)
axes.scatter(disc_3[1:-1], [1/3, 2/3], edgecolors=color_list[2], linestyle='--', facecolors='none', s=60)
axes.scatter(disc_4[1:-1], [1/3, 2/3], edgecolors=color_list[3], linestyle='--', facecolors='none', s=60)
axes.scatter(disc_5[1:-1], [1/3, 2/3], edgecolors=color_list[4], linestyle='--', facecolors='none', s=60)
axes.scatter(disc_6[1:-1], [1/3, 2/3], edgecolors=color_list[5], linestyle='--', facecolors='none', s=60)


axes.set_ylim([-0.0, 1.0])
axes.set_xlim([0, end_time])
axes.set_yticks([0, 1/3, 2/3, 1], [r'$0$',r'$\frac{1}{3}$', r'$\frac{2}{3}$', r'$1$'])
axes.grid(axis='y')
axes.set_xlabel('Days')
axes.set_ylabel('Survival Prob.')
plt.tight_layout()
plt.savefig(f'figs/illustration/iPET_step2.png', dpi=400, transparent=True)


probs_at_event_g1 = [pet_1, pet_2, pet_3]
probs_at_event_g2 = [pet_4, pet_5, pet_6]

hist_g1 = np.histogram(probs_at_event_g1, bins=[0, 1/3, 2/3, 1])
probs_g1 = hist_g1[0] / 3
cum_probs_g1 = np.cumsum(probs_g1)

hist_g2 = np.histogram(probs_at_event_g2, bins=[0, 1/3, 2/3, 1])
probs_g2 = hist_g2[0] / 3
cum_probs_g2 = np.cumsum(probs_g2)

cum_probs_g1 = np.insert(cum_probs_g1, 0, 0)
cum_probs_g2 = np.insert(cum_probs_g2, 0, 0)

fig2, axes2 = plt.subplots(nrows=1, ncols=2, figsize=(width_unit*0.5*1.2*2, height_unit*1.2), width_ratios=[1, 1.4])
axes2[0].hist([probs_at_event_g1, probs_at_event_g2], bins=[0, 1/3, 2/3, 1],
              color=[color_group[1], color_group[0]],
              rwidth=1, orientation='horizontal', stacked=True, )
axes2[0].set_ylim([-0.0, 1.0])
axes2[0].axvline(2, ls='dashed', c='grey', label='Perfect Cal.')
axes2[0].set_xlabel('Counts')
axes2[0].set_ylabel('Survival Prob.')
axes2[0].grid(axis='y')
axes2[0].set_xlim([0, 3.5])
axes2[0].set_xticks([0, 2])
axes2[0].set_yticks([0, 1/3, 2/3, 1], [r'$0$',r'$\frac{1}{3}$', r'$\frac{2}{3}$', r'$1$'])
# axes2[0].legend()

# axes2[1].plot([0, 1], [0, 1], ls='dashed', c='grey', label='Perfect Cal.')
axes2[1].plot([0, 1/3, 2/3, 1], cum_probs_g1, marker='o', markersize=10, c=color_group[1], linewidth=2,
              clip_on=False, zorder=10, label=r'$x=0$')
axes2[1].plot([0, 1/3, 2/3, 1], cum_probs_g2, marker='o', markersize=10, c=color_group[0], linewidth=2,
              clip_on=False, zorder=10, label=r'$x=1$')
axes2[1].plot([0, 1/3, 2/3, 1], 0.5 * (cum_probs_g1 + cum_probs_g2), ls='dashed', marker='o', markersize=10,
              c='grey', linewidth=2, clip_on=False, zorder=10, label='Marginal')
axes2[1].set_ylim([-0.0, 1.0])
axes2[1].set_xlim([-0.0, 1.0])
axes2[1].set_xticks([0, 1/3, 2/3, 1], [r'$0$', '1/3', '2/3', '1'])
axes2[1].set_yticks([0, 1/3, 2/3, 1], [r'$0$',r'$\frac{1}{3}$', r'$\frac{2}{3}$', r'$1$'])
axes2[1].set_xlabel('Predicted Prob.')
axes2[1].set_ylabel('Observed Prob.')
axes2[1].grid(False)
axes2[1].legend(framealpha=0.5).set_zorder(20)


plt.tight_layout()
plt.savefig(f'figs/illustration/iPET_hist.png', dpi=400, transparent=True)

#%% CSD method
last_time = end_time + 0.3

quantiles = [1/3, 2/3]
disc_1 = [dist_space[np.where(cdf_1>quan)[0][0]] for quan in quantiles]
disc_2 = [dist_space[np.where(cdf_2>quan)[0][0]] for quan in quantiles]
disc_3 = [dist_space[np.where(cdf_3>quan)[0][0]] for quan in quantiles]
disc_4 = [dist_space[np.where(cdf_4>quan)[0][0]] for quan in quantiles]
disc_5 = [dist_space[np.where(cdf_5>quan)[0][0]] for quan in quantiles]
disc_6 = [dist_space[np.where(cdf_6>quan)[0][0]] for quan in quantiles]
disc_1 = [0] + disc_1 + [last_time]
disc_2 = [0] + disc_2 + [last_time]
disc_3 = [0] + disc_3 + [last_time]
disc_4 = [0] + disc_4 + [last_time]
disc_5 = [0] + disc_5 + [last_time]
disc_6 = [0] + disc_6 + [last_time]

pet_1 = np.interp(e_1, disc_1[::], [1, 2/3, 1/3, 0])
pet_2 = np.interp(e_2, disc_2[::], [1, 2/3, 1/3, 0])
pet_3 = np.interp(e_3, disc_3[::], [1, 2/3, 1/3, 0])
pet_4 = np.interp(e_4, disc_4[::], [1, 2/3, 1/3, 0])
pet_5 = np.interp(e_5, disc_5[::], [1, 2/3, 1/3, 0])
pet_6 = np.interp(e_6, disc_6[::], [1, 2/3, 1/3, 0])

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(width_unit*0.75*1.2, height_unit*1.2))
axes.plot(disc_1, [1, 2/3, 1/3, 0], label='Patient A', c=color_group[1], zorder=10)
axes.plot(disc_2, [1, 2/3, 1/3, 0], label='Patient B', c=color_group[1], zorder=10)
axes.plot(disc_3, [1, 2/3, 1/3, 0], label='Patient C', c=color_group[1], zorder=10)
axes.plot(disc_4, [1, 2/3, 1/3, 0], label='Patient D', c=color_group[0], zorder=10)
axes.plot(disc_5, [1, 2/3, 1/3, 0], label='Patient E', c=color_group[0], zorder=10)
axes.plot(disc_6, [1, 2/3, 1/3, 0], label='Patient F', c=color_group[0], zorder=10)
axes.scatter(disc_1[1:-1], [2/3, 1/3], edgecolors=color_list[0], linestyle='--', facecolors='none', s=60)
axes.scatter(disc_2[1:-1], [2/3, 1/3], edgecolors=color_list[1], linestyle='--', facecolors='none', s=60)
axes.scatter(disc_3[1:-1], [2/3, 1/3], edgecolors=color_list[2], linestyle='--', facecolors='none', s=60)
axes.scatter(disc_4[1:-1], [2/3, 1/3], edgecolors=color_list[3], linestyle='--', facecolors='none', s=60)
axes.scatter(disc_5[1:-1], [2/3, 1/3], edgecolors=color_list[4], linestyle='--', facecolors='none', s=60)
axes.scatter(disc_6[1:-1], [2/3, 1/3], edgecolors=color_list[5], linestyle='--', facecolors='none', s=60)
axes.scatter(e_1, pet_1, marker='*', c=color_list[0], s=100, clip_on=False, zorder=20)
axes.scatter(e_2, pet_2, marker='*', c=color_list[1], s=100, clip_on=False, zorder=20)
axes.scatter(e_3, pet_3, marker='*', c=color_list[2], s=100, clip_on=False, zorder=20)
axes.scatter(e_4, pet_4, marker='*', c=color_list[3], s=100, clip_on=False, zorder=20)
axes.scatter(e_5, pet_5, marker='*', c=color_list[4], s=100, clip_on=False, zorder=20)
axes.scatter(e_6, pet_6, marker='*', c=color_list[5], s=100, clip_on=False, zorder=20)
axes.grid(axis='y')
axes.set_ylim([-0.0, 1.0])
axes.set_xlim([0, end_time])
axes.set_yticks([0, 1/3, 2/3, 1], [r'$0$',r'$\frac{1}{3}$', r'$\frac{2}{3}$', r'$1$'])
axes.set_xlabel('Days')
axes.set_ylabel('Survival Probability')
plt.tight_layout()
plt.savefig(f'figs/illustration/CSD_step1.png', dpi=400, transparent=True)


# conformal step

errors_33 = np.array([disc_1[1] - e_1, disc_2[1] - e_2, disc_3[1] - e_3, disc_4[1] - e_4, disc_5[1] - e_5, disc_6[1] - e_6])
errors_67 = np.array([disc_1[2] - e_1, disc_2[2] - e_2, disc_3[2] - e_3, disc_4[2] - e_4, disc_5[2] - e_5, disc_6[2] - e_6])
# errors_75 = np.array([disc_1[3] - e_1, disc_2[3] - e_2, disc_3[3] - e_3, disc_4[3] - e_4])

adj_33 = np.quantile(errors_33, 2/3)
adj_67 = np.quantile(errors_67, 1/3)
# adj_75 = np.quantile(errors_75, 0.25)

adj_disc_1 = [disc_1[0], disc_1[1] - adj_33, disc_1[2] - adj_67, last_time]
adj_disc_2 = [disc_2[0], disc_2[1] - adj_33, disc_2[2] - adj_67, last_time]
adj_disc_3 = [disc_3[0], disc_3[1] - adj_33, disc_3[2] - adj_67, last_time]
adj_disc_4 = [disc_4[0], disc_4[1] - adj_33, disc_4[2] - adj_67, last_time]
adj_disc_5 = [disc_5[0], disc_5[1] - adj_33, disc_5[2] - adj_67, last_time]
adj_disc_6 = [disc_6[0], disc_6[1] - adj_33, disc_6[2] - adj_67, last_time]

pet_1 = np.interp(e_1, adj_disc_1, [1, 2/3, 1/3, 0])
pet_2 = np.interp(e_2, adj_disc_2, [1, 2/3, 1/3, 0])
pet_3 = np.interp(e_3, adj_disc_3, [1, 2/3, 1/3, 0])
pet_4 = np.interp(e_4, adj_disc_4, [1, 2/3, 1/3, 0])
pet_5 = np.interp(e_5, adj_disc_5, [1, 2/3, 1/3, 0])
pet_6 = np.interp(e_6, adj_disc_6, [1, 2/3, 1/3, 0])

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(width_unit*0.75*1.2, height_unit*1.2))


axes.plot(adj_disc_1, [1, 2/3, 1/3, 0], label='Patient A', c=color_group[1], zorder=10)
axes.plot(adj_disc_2, [1, 2/3, 1/3, 0], label='Patient B', c=color_group[1], zorder=10)
axes.plot(adj_disc_3, [1, 2/3, 1/3, 0], label='Patient C', c=color_group[1], zorder=10)
axes.plot(adj_disc_4, [1, 2/3, 1/3, 0], label='Patient D', c=color_group[0], zorder=10)
axes.plot(adj_disc_5, [1, 2/3, 1/3, 0], label='Patient E', c=color_group[0], zorder=10)
axes.plot(adj_disc_6, [1, 2/3, 1/3, 0], label='Patient F', c=color_group[0], zorder=10)
axes.scatter(disc_1[1:-1], [2/3, 1/3,], edgecolors=color_list[0], linestyle='--', facecolors='none', s=60)
axes.scatter(disc_2[1:-1], [2/3, 1/3,], edgecolors=color_list[1], linestyle='--', facecolors='none', s=60)
axes.scatter(disc_3[1:-1], [2/3, 1/3,], edgecolors=color_list[2], linestyle='--', facecolors='none', s=60)
axes.scatter(disc_4[1:-1], [2/3, 1/3,], edgecolors=color_list[3], linestyle='--', facecolors='none', s=60)
axes.scatter(disc_5[1:-1], [2/3, 1/3,], edgecolors=color_list[4], linestyle='--', facecolors='none', s=60)
axes.scatter(disc_6[1:-1], [2/3, 1/3,], edgecolors=color_list[5], linestyle='--', facecolors='none', s=60)
axes.scatter(e_1, pet_1, marker='*', c=color_list[0], s=100, clip_on=False, zorder=20)
axes.scatter(e_2, pet_2, marker='*', c=color_list[1], s=100, clip_on=False, zorder=20)
axes.scatter(e_3, pet_3, marker='*', c=color_list[2], s=100, clip_on=False, zorder=20)
axes.scatter(e_4, pet_4, marker='*', c=color_list[3], s=100, clip_on=False, zorder=20)
axes.scatter(e_5, pet_5, marker='*', c=color_list[4], s=100, clip_on=False, zorder=20)
axes.scatter(e_6, pet_6, marker='*', c=color_list[5], s=100, clip_on=False, zorder=20)
axes.grid(axis='y')
axes.set_ylim([-0.0, 1.0])
axes.set_xlim([0, end_time])
axes.set_yticks([0, 1/3, 2/3, 1], [r'$0$',r'$\frac{1}{3}$', r'$\frac{2}{3}$', r'$1$'])
axes.set_xlabel('Days')
axes.set_ylabel('Survival Probability')
plt.tight_layout()
plt.savefig(f'figs/illustration/CSD_step2.png', dpi=400, transparent=True)

# histogram
probs_at_event_g1 = [pet_1, pet_2, pet_3]
probs_at_event_g2 = [pet_4, pet_5, pet_6]

hist_g1 = np.histogram(probs_at_event_g1, bins=[0, 1/3, 2/3, 1])
probs_g1 = hist_g1[0] / 3
cum_probs_g1 = np.cumsum(probs_g1)

hist_g2 = np.histogram(probs_at_event_g2, bins=[0, 1/3, 2/3, 1])
probs_g2 = hist_g2[0] / 3
cum_probs_g2 = np.cumsum(probs_g2)

cum_probs_g1 = np.insert(cum_probs_g1, 0, 0)
cum_probs_g2 = np.insert(cum_probs_g2, 0, 0)

fig2, axes2 = plt.subplots(nrows=1, ncols=2, figsize=(width_unit*0.5*1.2*2, height_unit*1.2), width_ratios=[1, 1.4])
axes2[0].hist([probs_at_event_g1, probs_at_event_g2], bins=[0, 1/3, 2/3, 1],
              color=[color_group[1], color_group[0]],
              rwidth=1, orientation='horizontal', stacked=True, )
axes2[0].set_ylim([-0.0, 1.0])
axes2[0].axvline(2, ls='dashed', c='grey', label='Perfect Cal.')
axes2[0].set_xlabel('Counts')
axes2[0].set_ylabel('Survival Prob.')
axes2[0].grid(axis='y')
axes2[0].set_xlim([0, 3.5])
axes2[0].set_xticks([0, 2])
axes2[0].set_yticks([0, 1/3, 2/3, 1], [r'$0$',r'$\frac{1}{3}$', r'$\frac{2}{3}$', r'$1$'])
# axes2[0].legend()

# axes2[1].plot([0, 1], [0, 1], ls='dashed', c='grey', label='Perfect Cal.')
axes2[1].plot([0, 1/3, 2/3, 1], cum_probs_g1, marker='o', markersize=10, c=color_group[1], linewidth=2,
              clip_on=False, zorder=10, label=r'$x=0$')
axes2[1].plot([0, 1/3, 2/3, 1], cum_probs_g2, marker='o', markersize=10, c=color_group[0], linewidth=2,
              clip_on=False, zorder=10, label=r'$x=1$')
axes2[1].plot([0, 1/3, 2/3, 1], 0.5 * (cum_probs_g1 + cum_probs_g2), ls='dashed', marker='o', markersize=10,
              c='grey', linewidth=2, clip_on=False, zorder=10, label='Marginal')
axes2[1].set_ylim([-0.0, 1.0])
axes2[1].set_xlim([-0.0, 1.0])
axes2[1].set_xticks([0, 1/3, 2/3, 1], [r'$0$', '1/3', '2/3', '1'])
axes2[1].set_yticks([0, 1/3, 2/3, 1], [r'$0$',r'$\frac{1}{3}$', r'$\frac{2}{3}$', r'$1$'])
axes2[1].set_xlabel('Predicted Prob.')
axes2[1].set_ylabel('Observed Prob.')
axes2[1].grid(False)
axes2[1].legend(framealpha=0.5).set_zorder(20)


plt.tight_layout()
plt.savefig(f'figs/illustration/CSD_hist.png', dpi=400, transparent=True)
