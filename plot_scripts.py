#%% Plot the KM curves and the histogram for all the datasets
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

from data import make_survival_data
from SurvivalEVAL.Evaluations.util import KaplanMeier

sns.set(style="whitegrid")
colors = sns.color_palette("deep")

datasets = [
    "VALCT", "DLBCL", "PBC", "GBM", "NACD", "GBSG", "METABRIC", "SUPPORT",
    "AIDS", "HFCR", "WPBC", "BMT", "churn", "credit", "employee", "PDM",
    "MIMIC-IV_hosp", "MIMIC-IV_all",
    "DBCD", "FLCHAIN", "NWTCO", "NPC", "WHAS", "WHAS500",
    "SEER_liver", "SEER_lung", "SEER_prostate", "SEER_brain", "SEER_thyroid",
    "SEER_stomach", "SEER_urinary", "SEER_kidney", "SEER_breast",
]
for data_name in datasets:
    data, _ = make_survival_data(data_name)
    data = data.astype({'time': 'float64', 'event': 'int32'})
    censor_rate = 1 - data.event.mean()

    event_times = data.time.values[data.event.values == 1]
    censor_times = data.time.values[data.event.values == 0]

    # Sturges formula
    intervals = math.ceil(math.log2(data.shape[0]) + 1)
    bins = np.linspace(0, round(data.time.max()), intervals)

    fig, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))

    km_estimator = KaplanMeier(data.time.values, data.event.values)
    survival_times = km_estimator.survival_times
    survival_probabilities = km_estimator.survival_probabilities
    print(f"{data_name}; last time: {survival_times[-1]}; last survival probability: {survival_probabilities[-1]}")
    if survival_times[0] != 0:
        survival_times = np.insert(survival_times, 0, 0)
        survival_probabilities = np.insert(survival_probabilities, 0, 1.0)
    ax0.step(survival_times, survival_probabilities, linewidth=2.5, color=colors[0],
             clip_on=False, zorder=3)
    # ax0.set_title("Kaplan-Meier Curve")
    ax0.set_ylabel("Survival Probability", color=colors[0], weight='bold')
    ax0.set_xlabel("Time", weight='bold')
    ax0.set_ylim([0, 1.05])
    ax0.tick_params(axis='y', colors=colors[0])
    ax0.set_xlim([0, max(survival_times)])
    ax0.xaxis.grid(False)
    # ax0.set_xticks([])
    xmin, xmax = ax0.get_xlim()

    ax1 = ax0.twinx()
    ax1.hist([event_times, censor_times], bins=bins, histtype='barstacked', stacked=True, alpha=0.9, color=[colors[2], colors[1]], zorder=2)
    # ax1.set_yscale('log')
    ax1.set_ylabel('Counts', color='black', weight='bold')
    ax1.legend(['Event', 'Censored'], loc='best')
    ax1.yaxis.grid(False)
    ax0.set_zorder(ax1.get_zorder() + 1)
    ax0.patch.set_visible(False)

    # ax1.set_title("Event/Censor Time Histogram")

    # fig.set_size_inches(12, 12)
    # plt.suptitle(
    #     '{}\n #Subjects: {}; %Censoring: {:.1f}%'.format(data_rename[data_name], data.shape[0], round(censor_rate * 100, 3))
    # )
    # plt.suptitle(
    #     '{}'.format(data_rename[data_name])
    # )
    # plt.show()
    plt.tight_layout()
    fig.savefig(f'figs/data/{data_name}.png', dpi=300)
    plt.close(fig)


#%% plot main results
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os

sns.set_theme(style="whitegrid")

# models = ['AFT', 'GB', 'MTLR', 'CoxPH', 'DeepHit', 'CoxTime', 'CQRNN']
model_name = ['AFT', 'GB', 'N-MTLR', 'DeepSurv', 'DeepHit', 'CoxTime', 'CQRNN']

datasets = ['churn', 'employee', 'FLCHAIN', 'GBM', 'GBSG', 'HFCR', 'METABRIC', 'MIMIC-IV_all',
            'NACD', 'PBC', 'PDM', 'SUPPORT', 'WHAS500',
            'SEER_liver', 'SEER_brain', 'SEER_stomach', ]
metrics_names = ['Cindex', 'IBS', 'KM_cal', 'MAE_PO', 'xCal_stats', 'wsc_xCal_stats']


for data in datasets:
    models = pd.read_csv(f'results/main/{data}.csv')
    result_path = f'runs/{data}'

    km = models[models['Model'] == 'KM']
    km_results = np.load(f'{result_path}/KM/{km.Timestamp.values[0]}/performance.pkl', allow_pickle=True)

    comparisions = models[models['Model'] != 'KM']

    path = f'figs/main/{data}'
    os.makedirs(path, exist_ok=True)

    # if data == "MIMIC-IV_all":
    #     comparisions = comparisions[comparisions['Model'] != 'CQRNN']
    #     position = np.array([0, 1, 2,
    #                          4, 5, 6,
    #                          8, 9, 10,
    #                          12, 13, 14,
    #                          16, 17, 18,
    #                          20, 21, 22,])
    #     model_name = ['AFT', 'GB', 'N-MTLR', 'DeepSurv', 'DeepHit', 'CoxTime']
    # else:
    position = np.array([0, 1, 2,
                         4, 5, 6,
                         8, 9, 10,
                         12, 13, 14,
                         16, 17, 18,
                         20, 21, 22,
                         24, 25, 26,])
    model_name = ['AFT', 'GB', 'N-MTLR', 'DeepSurv', 'DeepHit', 'CoxTime', 'CQRNN']

    for metric in metrics_names:
        performance = []
        for compare in comparisions.itertuples():
            timestamp = compare.Timestamp
            model = compare.Model
            results = np.load(f'{result_path}/{model}/{timestamp}/performance.pkl', allow_pickle=True)
            performance.append(results[metric])

        fig, ax = plt.subplots(figsize=(3.5, 1.5))

        if data != 'MIMIC-IV_all':
            # violin plot with colors, 0, 3, 6, 9 ... have same color, 1, 4, 7, 10 ... have same color, etc.
            ax.violinplot(performance, positions=position, widths=0.7, points=10,
                          showmeans=True, showextrema=False, showmedians=False)
        else:
            # drop the last three position for MIMIC-IV_all
            ax.violinplot(performance[:-3], positions=position[:-3], widths=0.7, points=10,
                          showmeans=True, showextrema=False, showmedians=False)


        # set colors
        for i, pc in enumerate(ax.collections):
            pc.set_facecolor(sns.color_palette("deep")[i % 3])
            pc.set_edgecolor('black')
            pc.set_alpha(0.8)
            pc.set_linewidth(1)
            pc.set_zorder(5)

        # # calculate 25% and 75% percentiles
        # percentiles = np.percentile(performance, [25, 75], axis=1)
        # ax.vlines(position, percentiles[0], percentiles[1], color='k', linestyle='-', lw=1, zorder=8)
        #
        # # calculate mean
        # means = np.mean(performance, axis=1)
        # ax.scatter(position, means, marker='_', color='k', s=5, zorder=10)

        # ax.errorbar(x - width/2, bar_means['Non-CSD'], yerr=bar_ci['Non-CSD'], fmt='o', capsize=5, label='Non-CSD')
        # ax.errorbar(x + width/2, bar_means['CSD'], yerr=bar_ci['CSD'], fmt='o', capsize=5, label='CSD')
        if metric == 'KM_cal' or metric == 'xCal_stats':
            ax.hlines(np.array(km_results[metric]).mean(), xmin=-1.5, xmax=position.max() + 0.5, colors='r',
                      linestyles='dashed', zorder=3)

        ax.set_xlim([-0.5, position.max() + 0.5])
        ax.grid(axis='x')
        # no x labels
        ax.set_xticks([])
        # with x labels
        # ax.set_xticks(position[::3] + 1)
        # ax.set_xticklabels(model_name, fontsize=9)
        # ax.legend(loc='upper center', ncols=2)
        # if metric == 'Concordance':
        #     ax.set_ylim([0.5, 1])
        if metric == 'KM_cal' or metric == 'xCal_stats' or metric == 'wsc_xCal_stats':
            plt.yscale('log')

        fig.tight_layout()
        plt.subplots_adjust(left=0.15, right=0.99, top=0.97, bottom=0.03)
        plt.setp(ax.get_xticklabels(), rotation=25, horizontalalignment='center')
        fig.savefig(f'{path}/{metric}.png', dpi=400, transparent=True)
        plt.close(fig)


#%% Count the number of won/lost/draws for each dataset
from scipy.stats import ttest_ind_from_stats, ttest_ind
import pandas as pd
import numpy as np

datasets = ['churn', 'employee', 'FLCHAIN', 'GBM', 'GBSG', 'HFCR', 'MIMIC-IV_all',
            'NACD', 'PBC', 'PDM', 'SUPPORT', 'WHAS500',
            'SEER_liver', 'SEER_brain', 'SEER_stomach', ]
metrics_names = ['Cindex', 'IBS', 'KM_cal', 'MAE_PO', 'xCal_stats', 'wsc_xCal_stats']


for metric in metrics_names:
    total_bl = 0
    lower_bl = 0
    lower_significant_bl = 0
    higher_bl = 0
    higher_significant_bl = 0
    draw_bl = 0

    total_csd = 0
    lower_csd = 0
    lower_significant_csd = 0
    higher_csd = 0
    higher_significant_csd = 0
    draw_csd = 0

    for data in datasets:
        models = pd.read_csv(f'results/main/{data}.csv')
        result_path = f'runs/{data}'

        comparisions = models[models['Model'] != 'KM']

        if data == 'MIMIC-IV_all':
            # No CQRNN results for MIMIC-IV_all
            comparisions = comparisions[comparisions['Model'] != 'CQRNN']
            model_name = ['AFT', 'GB', 'MTLR', 'CoxPH', 'DeepHit', 'CoxTime']
        else:
            model_name = ['AFT', 'GB', 'MTLR', 'CoxPH', 'DeepHit', 'CoxTime', 'CQRNN']

        baselines = comparisions[comparisions['Method'] == 'Baseline']
        CSD = comparisions[comparisions['Method'] == 'CSD']
        CSDiPET = comparisions[comparisions['Method'] == 'CSD-iPET']

        for model in model_name:
            total_bl += 1
            total_csd += 1

            baseline = baselines[baselines['Model'] == model]
            csd = CSD[CSD['Model'] == model]
            csdipet = CSDiPET[CSDiPET['Model'] == model]

            bl_results = np.load(f'{result_path}/{model}/{baseline.Timestamp.values[0]}/performance.pkl', allow_pickle=True)[metric]
            csd_results = np.load(f'{result_path}/{model}/{csd.Timestamp.values[0]}/performance.pkl', allow_pickle=True)[metric]
            csdipet_results = np.load(f'{result_path}/{model}/{csdipet.Timestamp.values[0]}/performance.pkl', allow_pickle=True)[metric]

            bl_mean = np.format_float_positional(sum(bl_results) / len(bl_results), precision=3, unique=False, fractional=False)
            csd_mean = np.format_float_positional(sum(csd_results) / len(csd_results), precision=3, unique=False, fractional=False)
            csdipet_mean = np.format_float_positional(sum(csdipet_results) / len(csdipet_results), precision=3, unique=False, fractional=False)

            if bl_mean == csdipet_mean:
                draw_bl += 1
            elif bl_mean < csdipet_mean:
                higher_bl += 1
                p = ttest_ind(bl_results, csdipet_results, equal_var=False)[1]
                if p < 0.05:
                    higher_significant_bl += 1
            else:
                lower_bl += 1
                p = ttest_ind(bl_results, csdipet_results, equal_var=False)[1]
                if p < 0.05:
                    lower_significant_bl += 1

            if csd_mean == csdipet_mean:
                draw_csd += 1
            elif csd_mean < csdipet_mean:
                higher_csd += 1
                p = ttest_ind(csd_results, csdipet_results, equal_var=False)[1]
                if p < 0.05:
                    higher_significant_csd += 1
            else:
                lower_csd += 1
                p = ttest_ind(csd_results, csdipet_results, equal_var=False)[1]
                if p < 0.05:
                    lower_significant_csd += 1
    print('*************************Compare with Baselines*************************')
    print(f'{metric}:')
    print(f'Lower: {lower_bl}, Lower Significant: {lower_significant_bl}')
    print(f'Higher: {higher_bl}, Higher Significant: {higher_significant_bl}')
    print(f'Draw: {draw_bl}')
    print(f'Total: {lower_bl + higher_bl + draw_bl}')
    print(f'Total: {total_bl}')
    print('-------------------------')
    print('*************************Compare with CSD*************************')
    print(f'{metric}:')
    print(f'Lower: {lower_csd}, Lower Significant: {lower_significant_csd}')
    print(f'Higher: {higher_csd}, Higher Significant: {higher_significant_csd}')
    print(f'Draw: {draw_csd}')
    print(f'Total: {lower_csd + higher_csd + draw_csd}')
    print(f'Total: {total_csd}')
    print('-------------------------')


#%% Plot computational time analysis
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os

sns.set_theme(style="whitegrid")

# models = ['AFT', 'GB', 'MTLR', 'CoxPH', 'DeepHit', 'CoxTime', 'CQRNN']
model_name = ['AFT', 'GB', 'N-MTLR', 'DeepSurv', 'DeepHit', 'CoxTime', 'CQRNN']

datasets = ['churn', 'employee', 'FLCHAIN', 'GBM', 'GBSG', 'HFCR', 'MIMIC-IV_all',
            'NACD', 'PBC', 'PDM', 'SUPPORT', 'WHAS500',
            'SEER_liver', 'SEER_brain', 'SEER_stomach', ]
metrics_names = ['train_times', 'infer_times']
width = 0.15  # the width of the bars

for data in datasets:
    comparisions = pd.read_csv(f'results/runtime/{data}.csv')
    result_path = f'runs/{data}'

    path = f'figs/runtime/{data}'
    os.makedirs(path, exist_ok=True)

    position = np.array([0, 1, 2,
                         4, 5, 6,
                         8, 9, 10,
                         12, 13, 14,
                         16, 17, 18,
                         20, 21, 22,
                         24, 25, 26,])
    model_name = ['AFT', 'GB', 'N-MTLR', 'DeepSurv', 'DeepHit', 'CoxTime', 'CQRNN']

    for metric in metrics_names:
        performance = []
        for compare in comparisions.itertuples():
            timestamp = compare.Timestamp
            model = compare.Model
            results = np.load(f'{result_path}/{model}/{timestamp}/performance.pkl', allow_pickle=True)
            try:
                performance.append(results[metric])
            except KeyError:
                performance.append(results["cal_times"])

        fig, ax = plt.subplots(figsize=(2.5, 1.5))

        means = np.mean(performance, axis=1)
        stds = np.std(performance, axis=1)
        confidences = 1.96 * stds / np.sqrt(len(performance[0])) # 95% confidence interval

        if data != 'MIMIC-IV_all':
            # box plot with colors, 0, 3, 6, 9 ... have same color, 1, 4, 7, 10 ... have same color, etc.
            # bplot = ax.boxplot(performance, positions=position, widths=0.7, patch_artist=True, meanline=True, showmeans=True, showfliers=False)
            ax.errorbar(position[0::3] - width, means[0::3], yerr=confidences[0::3], fmt='o', capsize=5,
                        c='#708ec0',
                        label='Baseline')
            ax.errorbar(position[1::3], means[1::3], yerr=confidences[1::3], fmt='o', capsize=5,
                        c='#e49d75',
                        label='CSD')
            ax.errorbar(position[2::3] + width, means[2::3], yerr=confidences[2::3], fmt='o', capsize=5,
                        c='#77b986',
                        label='CSD-iPOT')
        else:
            # drop the last three position for MIMIC-IV_all
            # bplot = ax.boxplot(performance[:-3], positions=position[:-3], widths=0.7, patch_artist=True,
            #            meanline=True, showmeans=True, showfliers=False)
            ax.errorbar(position[0:-3:3] - width, means[0:-3:3], yerr=confidences[0:-3:3], fmt='o', capsize=5,
                        c='#708ec0',
                        label='Baseline')
            ax.errorbar(position[1:-3:3], means[1:-3:3], yerr=confidences[1:-3:3], fmt='o', capsize=5,
                        c='#e49d75',
                        label='CSD')
            ax.errorbar(position[2:-3:3] + width, means[2:-3:3], yerr=confidences[2:-3:3], fmt='o', capsize=5,
                        c='#77b986',
                        label='CSD-iPOT')
        # colors = sns.color_palette("deep")[:3] * 7
        # # set colors
        # # for i, pc in enumerate(ax.artists):
        # #     pc.set_facecolor(sns.color_palette("deep")[i % 3])
        # #     pc.set_edgecolor('black')
        # #     pc.set_alpha(0.8)
        # #     pc.set_linewidth(1)
        # #     pc.set_zorder(5)
        # for patch, color in zip(bplot['boxes'], colors):
        #     patch.set_facecolor(color)
        #     patch.set_edgecolor(color)
        #     # patch.set_alpha(0.8)
        #     patch.set_linewidth(0.1)
        #     patch.set_zorder(5)
        #
        # for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        #     plt.setp(bplot[element], color=colors)
        #
        # for median in bplot['medians']:
        #     median.set(color='black', linewidth=1, zorder=10)


        ax.set_xlim([-0.5 - 4 * width, position.max() + 0.5 + 4 * width])
        ax.grid(axis='x')
        # no x labels
        ax.set_xticks([])
        # with x labels
        # ax.set_xticks(position[::3] + 1)
        # ax.set_xticklabels(model_name, fontsize=9)
        # ax.legend(loc='upper center', ncols=2)
        # plt.yscale('log')

        fig.tight_layout()
        plt.subplots_adjust(left=0.17, right=0.99, top=0.97, bottom=0.03)
        plt.setp(ax.get_xticklabels(), rotation=25, horizontalalignment='center')
        fig.savefig(f'{path}/{metric}.png', dpi=400, transparent=True)
        plt.close(fig)


#%% ablation #1, number of sampling
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os

sns.set_theme(style="whitegrid")

colors = sns.color_palette("Set2", 5)

# models = ['AFT', 'GB', 'MTLR', 'CoxPH', 'DeepHit', 'CoxTime', 'CQRNN']
model_name = ['AFT', 'GB', 'N-MTLR', 'DeepSurv', 'DeepHit', 'CoxTime', 'CQRNN']

datasets = ['churn', 'employee', 'FLCHAIN', 'GBM', 'GBSG', 'HFCR', 'MIMIC-IV_all',
            'NACD', 'PBC', 'PDM', 'SUPPORT', 'WHAS500',
            'SEER_liver', 'SEER_brain', 'SEER_stomach', ]
metrics_names = ['Cindex', 'IBS', 'KM_cal', 'MAE_PO', 'xCal_stats', 'wsc_xCal_stats']


for data in datasets:
    comparisions = pd.read_csv(f'results/sampling/{data}.csv')
    result_path = f'runs/{data}'

    path = f'figs/sampling/{data}'
    os.makedirs(path, exist_ok=True)

    position = np.array([0, 1, 2, 3, 4,
                         6, 7, 8, 9, 10,
                        12, 13, 14, 15, 16,
                        18, 19, 20, 21, 22,
                        24, 25, 26, 27, 28,
                        30, 31, 32, 33, 34,
                        36, 37, 38, 39, 40,])
    # position = np.array([0, 1, 2, 3,
    #                      6, 7, 8, 9,
    #                     12, 13, 14, 15,
    #                     18, 19, 20, 21,
    #                     24, 25, 26, 27,
    #                     30, 31, 32, 33,
    #                     36, 37, 38, 39,])

    model_name = ['AFT', 'GB', 'N-MTLR', 'DeepSurv', 'DeepHit', 'CoxTime', 'CQRNN']

    for metric in metrics_names:
        performance = []
        for compare in comparisions.itertuples():
            timestamp = compare.Timestamp
            model = compare.Model
            results = np.load(f'{result_path}/{model}/{timestamp}/performance.pkl', allow_pickle=True)
            performance.append(results[metric])

        fig, ax = plt.subplots(figsize=(3.5, 1.5))


        if data != 'MIMIC-IV_all':
            # violin plot with colors, 0, 3, 6, 9 ... have same color, 1, 4, 7, 10 ... have same color, etc.
            ax.violinplot(performance, positions=position, widths=0.7, points=10,
                          showmeans=True, showextrema=False, showmedians=False)
        else:
            # drop the last three position for MIMIC-IV_all
            ax.violinplot(performance[:-5], positions=position[:-5], widths=0.7, points=10,
                          showmeans=True, showextrema=False, showmedians=False)


        # set colors
        for i, pc in enumerate(ax.collections):
            pc.set_facecolor(colors[i % 5])
            pc.set_edgecolor('black')
            pc.set_alpha(0.8)
            pc.set_linewidth(1)
            pc.set_zorder(5)


        ax.set_xlim([-0.5, position.max() + 0.5])
        ax.grid(axis='x')
        # no x labels
        ax.set_xticks([])
        # with x labels
        # ax.set_xticks(position[::5] + 1)
        # ax.set_xticklabels(model_name, fontsize=9)
        # ax.legend(loc='upper center', ncols=2)
        # if metric == 'Concordance':
        #     ax.set_ylim([0.5, 1])
        if metric == 'KM_cal' or metric == 'xCal_stats' or metric == 'wsc_xCal_stats':
            plt.yscale('log')

        fig.tight_layout()
        plt.subplots_adjust(left=0.15, right=0.99, top=0.97, bottom=0.03)
        plt.setp(ax.get_xticklabels(), rotation=25, horizontalalignment='center')
        fig.savefig(f'{path}/{metric}.png', dpi=400, transparent=False)
        plt.close(fig)


#%% ablation #2, number of percentiles
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os

sns.set_theme(style="whitegrid")

# color_map = sns.dark_palette("#69d", as_cmap=True, reverse=False)
# grid = int(256/4)
# colors = [color_map(i) for i in range(0, 256, grid)]
# colors = colors * 7
colors = sns.color_palette("Set2", 4)

# models = ['AFT', 'GB', 'MTLR', 'CoxPH', 'DeepHit', 'CoxTime', 'CQRNN']
model_name = ['AFT', 'GB', 'N-MTLR', 'DeepSurv', 'DeepHit', 'CoxTime', 'CQRNN']

datasets = ['churn', 'employee', 'FLCHAIN', 'GBM', 'GBSG', 'HFCR', 'MIMIC-IV_all',
            'NACD', 'PBC', 'PDM', 'SUPPORT', 'WHAS500',
            'SEER_liver', 'SEER_brain', 'SEER_stomach', ]
metrics_names = ['Cindex', 'IBS', 'KM_cal', 'MAE_PO', 'xCal_stats', 'wsc_xCal_stats']


for data in datasets:
    comparisions = pd.read_csv(f'results/quantile/{data}.csv')
    result_path = f'runs/{data}'

    path = f'figs/quantile/{data}'
    os.makedirs(path, exist_ok=True)

    # position = np.array([0, 1, 2, 3, 4,
    #                      6, 7, 8, 9, 10,
    #                     12, 13, 14, 15, 16,
    #                     18, 19, 20, 21, 22,
    #                     24, 25, 26, 27, 28,
    #                     30, 31, 32, 33, 34,
    #                     36, 37, 38, 39, 40,])
    position = np.array([0, 1, 2, 3,
                         6, 7, 8, 9,
                        12, 13, 14, 15,
                        18, 19, 20, 21,
                        24, 25, 26, 27,
                        30, 31, 32, 33,
                        36, 37, 38, 39,])

    model_name = ['AFT', 'GB', 'N-MTLR', 'DeepSurv', 'DeepHit', 'CoxTime', 'CQRNN']

    for metric in metrics_names:
        performance = []
        for compare in comparisions.itertuples():
            timestamp = compare.Timestamp
            model = compare.Model
            if timestamp is not np.nan:
                results = np.load(f'{result_path}/{model}/{timestamp}/performance.pkl', allow_pickle=True)
                performance.append(results[metric])
            else:
                pass

        fig, ax = plt.subplots(figsize=(3.5, 1.5))


        if data != 'MIMIC-IV_all':
            # violin plot with colors, 0, 3, 6, 9 ... have same color, 1, 4, 7, 10 ... have same color, etc.
            ax.violinplot(performance, positions=position, widths=0.8, points=10,
                          showmeans=True, showextrema=False, showmedians=False)
        else:
            # drop the last three position for MIMIC-IV_all
            ax.violinplot(performance[:-5], positions=position[:-5], widths=0.8, points=10,
                          showmeans=True, showextrema=False, showmedians=False)


        # set colors
        for i, pc in enumerate(ax.collections):
            pc.set_facecolor(colors[i % 4])
            pc.set_edgecolor('black')
            pc.set_alpha(0.8)
            pc.set_linewidth(1)
            pc.set_zorder(5)


        ax.set_xlim([-0.5, position.max() + 0.5])
        ax.grid(axis='x')
        # no x labels
        ax.set_xticks([])
        # with x labels
        # ax.set_xticks(position[::5] + 1)
        # ax.set_xticklabels(model_name, fontsize=9)
        # ax.legend(loc='upper center', ncols=2)
        # if metric == 'Concordance':
        #     ax.set_ylim([0.5, 1])
        if metric == 'KM_cal' or metric == 'xCal_stats' or metric == 'wsc_xCal_stats':
            plt.yscale('log')

        fig.tight_layout()
        plt.subplots_adjust(left=0.15, right=0.99, top=0.97, bottom=0.03)
        plt.setp(ax.get_xticklabels(), rotation=25, horizontalalignment='center')
        fig.savefig(f'{path}/{metric}.png', dpi=400, transparent=False)
        plt.close(fig)


#%% DeepHit and CQRNN on FLCHAIN
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
from lifelines import KaplanMeierFitter


custom_params = {"axes.spines.right": False, "axes.spines.top": False,}
plt.rcParams['figure.constrained_layout.use'] = True
sns.set_theme(style="white", rc=custom_params)

plt.rc('axes', titlesize=10)     # fontsize of the axes title
plt.rc('axes', labelsize=10)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
plt.rc('legend', fontsize=8)    # legend fontsize
plt.rc('figure', titlesize=10)  # fontsize of the figure title
height_unit = 2.8
width_unit = 2.8
plt.rcParams['font.size'] = 12


sns.set_theme(style="whitegrid")
colors = sns.color_palette("Set2", 5)

t_true = np.load('temp/deephit/t_test.npy', allow_pickle=True)
e_true = np.load('temp/deephit/e_test.npy', allow_pickle=True)
km = KaplanMeierFitter().fit(t_true, e_true)

# load results
deephit_bl_dcal = np.load('temp/deephit/dcal_baseline.npy', allow_pickle=True)
deephit_csd_dcal = np.load('temp/deephit/dcal_csd.npy', allow_pickle=True)
deephit_csdipet_dcal = np.load('temp/deephit/dcal_csdipot.npy', allow_pickle=True)

deephit_bl_slope = deephit_bl_dcal.cumsum() / deephit_bl_dcal.sum()
deephit_csd_slope = deephit_csd_dcal.cumsum() / deephit_csd_dcal.sum()
deephit_csdipet_slope = deephit_csdipet_dcal.cumsum() / deephit_csdipet_dcal.sum()

deephit_bl_slope = np.insert(deephit_bl_slope, 0, 0)
deephit_csd_slope = np.insert(deephit_csd_slope, 0, 0)
deephit_csdipet_slope = np.insert(deephit_csdipet_slope, 0, 0)

deephit_bl_pred = np.load('temp/deephit/surv_test_baseline.npy', allow_pickle=True)
deephit_bl_times = np.load('temp/deephit/time_coordinates_baseline.npy', allow_pickle=True)
deephit_csd_pred = np.load('temp/deephit/quan_pred_csd.npy', allow_pickle=True)
deephit_csd_levels = np.load('temp/deephit/quan_levels_csd.npy', allow_pickle=True)
deephit_csdipot_pred = np.load('temp/deephit/quan_pred_csdipot.npy', allow_pickle=True)
deephit_csdipot_levels = np.load('temp/deephit/quan_levels_csdipot.npy', allow_pickle=True)

# cqrnn_baselines_dcal = np.load('temp/cqrnn/dcal_baseline.npy', allow_pickle=True)
# cqrnn_csd_dcal = np.load('temp/cqrnn/dcal_csd.npy', allow_pickle=True)
# cqrnn_csdipet_dcal = np.load('temp/cqrnn/dcal_csdipot.npy', allow_pickle=True)
#
# cqrnn_bl_slope = cqrnn_baselines_dcal.cumsum() / cqrnn_baselines_dcal.sum()
# cqrnn_csd_slope = cqrnn_csd_dcal.cumsum() / cqrnn_csd_dcal.sum()
# cqrnn_csdipet_slope = cqrnn_csdipet_dcal.cumsum() / cqrnn_csdipet_dcal.sum()
#
# cqrnn_bl_slope = np.insert(cqrnn_bl_slope, 0, 0)
# cqrnn_csd_slope = np.insert(cqrnn_csd_slope, 0, 0)
# cqrnn_csdipet_slope = np.insert(cqrnn_csdipet_slope, 0, 0)
#
# cqrnn_bl_pred = np.load('temp/cqrnn/quan_pred_baseline.npy', allow_pickle=True)
# cqrnn_bl_levels = np.load('temp/cqrnn/quan_levels_baseline.npy', allow_pickle=True)
# cqrnn_csd_pred = np.load('temp/cqrnn/quan_pred_csd.npy', allow_pickle=True)
# cqrnn_csd_levels = np.load('temp/cqrnn/quan_levels_csd.npy', allow_pickle=True)
# cqrnn_csdipot_pred = np.load('temp/cqrnn/quan_pred_csdipot.npy', allow_pickle=True)
# cqrnn_csdipot_levels = np.load('temp/cqrnn/quan_levels_csdipot.npy', allow_pickle=True)


fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(width_unit * 4, height_unit ))

max_time = km.survival_function_.index.max()
# DeepHit baselines
ax[0].step(km.survival_function_.index, km.survival_function_.KM_estimate, color=colors[-1], linestyle='--',)
for i in [0, 1, 10, 100]:
    ax[0].plot(deephit_bl_times, deephit_bl_pred[i], label=f'{i}',  color=colors[0])
ax[0].set_ylabel('Survival Probability')
ax[0].set_xlabel('Time')
ax[0].set_xlim([0, max_time])
ax[0].set_ylim([0, 1.0])
# ax[0, 0].legend()

# DeepHit CSD
ax[1].step(km.survival_function_.index, km.survival_function_.KM_estimate, color=colors[-1], linestyle='--',)
for i in [0, 1, 10, 100]:
    ax[1].plot(deephit_csd_pred[i], 1- deephit_csd_levels, label=f'{i}',  color=colors[1])
ax[1].set_ylabel('Survival Probability')
ax[1].set_xlabel('Time')
ax[1].set_xlim([0, max_time])
ax[1].set_ylim([0, 1.0])

# DeepHit CSD-iPET
ax[2].step(km.survival_function_.index, km.survival_function_.KM_estimate, color=colors[-1], linestyle='--',)
for i in [0, 1, 10, 100]:
    ax[2].plot(deephit_csdipot_pred[i], 1 - deephit_csdipot_levels, label=f'{i}',  color=colors[2])
ax[2].set_ylabel('Survival Probability')
ax[2].set_xlabel('Time')
ax[2].set_xlim([0, max_time])
ax[2].set_ylim([0, 1.0])

# dcal plot
observed = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) / 10
# ax[3].plot([0, 1], [0, 1], ls='dashed', c='grey', label='Perfect Cal.')
ax[3].plot(observed, deephit_bl_slope, marker='o', markersize=10, c=colors[0], linewidth=2, label='Baseline',
              clip_on=False, zorder=10)
ax[3].plot(observed, deephit_csd_slope, marker='o', markersize=10, c=colors[1], linewidth=2, label='CSD',
              clip_on=False, zorder=10)
ax[3].plot(observed, deephit_csdipet_slope, marker='o', markersize=10, c=colors[2], linewidth=2, label='CSD-iPOT',
              clip_on=False, zorder=10)
ax[3].set_xlim([0, 1])
ax[3].set_ylim([0, 1])
ax[3].set_xlabel('Predicted Probability')
ax[3].set_ylabel('Observed Probability')
ax[3].legend(loc='upper left')



# # CQRNN baselines
# ax[1, 0].step(km.survival_function_.index, km.survival_function_.KM_estimate, color=colors[-1])
# for i in [0, 1, 10, 100]:
#     ax[1, 0].step(cqrnn_bl_pred[i], 1- cqrnn_bl_levels, label=f'{i}', linestyle='--', color=colors[0])
# ax[1, 0].set_ylabel('Survival Probability')
# ax[1, 0].set_xlabel('Time')
# ax[1, 0].set_xlim([0, max_time])
# ax[1, 0].set_ylim([0, 1.05])
#
# # CQRNN CSD
# ax[1, 1].step(km.survival_function_.index, km.survival_function_.KM_estimate, color=colors[-1])
# for i in [0, 1, 10, 100]:
#     ax[1, 1].step(cqrnn_csd_pred[i], 1 - cqrnn_csd_levels, label=f'{i}', linestyle='--', color=colors[1])
# ax[1, 1].set_ylabel('Survival Probability')
# ax[1, 1].set_xlabel('Time')
# ax[1, 1].set_xlim([0, max_time])
# ax[1, 1].set_ylim([0, 1.05])
#
# # CQRNN CSD-iPET
# ax[1, 2].step(km.survival_function_.index, km.survival_function_.KM_estimate, color=colors[-1])
# for i in [0, 1, 10, 100]:
#     ax[1, 2].step(cqrnn_csdipot_pred[i], 1 - cqrnn_csdipot_levels, label=f'{i}', linestyle='--', color=colors[2])
# ax[1, 2].set_ylabel('Survival Probability')
# ax[1, 2].set_xlabel('Time')
# ax[1, 2].set_xlim([0, max_time])
# ax[1, 2].set_ylim([0, 1.05])
#
# # dcal plot
# ax[1, 3].plot([0, 1], [0, 1], ls='dashed', c='grey', label='Perfect Cal.')
# ax[1, 3].plot(observed, cqrnn_bl_slope, marker='o', markersize=10, c=colors[0], linewidth=2, label='Baseline',
#               clip_on=False, zorder=10)
# ax[1, 3].plot(observed, cqrnn_csd_slope, marker='o', markersize=10, c=colors[1], linewidth=2, label='CSD',
#                 clip_on=False, zorder=10)
# ax[1, 3].plot(observed, cqrnn_csdipet_slope, marker='o', markersize=10, c=colors[2], linewidth=2, label='CSD-iPET',
#                 clip_on=False, zorder=10)
# ax[1, 3].set_xlim([0, 1])
# ax[1, 3].set_ylim([0, 1])
# ax[1, 3].set_xlabel('Predicted Prob.')
# ax[1, 3].set_ylabel('Observed Prob.')
# ax[1, 3].legend(loc='upper left')


plt.tight_layout()
plt.savefig(f'figs/example/deephit_cqrnn.png', dpi=400, transparent=True)




