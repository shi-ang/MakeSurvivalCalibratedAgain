#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from SurvivalEVAL.Evaluations.util import KaplanMeier, quantile_to_survival


CSD_AFT_HFCR = [
    # age, old, then young
    [0.06941201, 1.42632386, 0.60454144, 0.8384018 , 1.95287313,
     2.02168955, 2.24489916, 1.28061969, 1.28061969, 1.28061969],   # 0.009510205657545593
    [1.10377975, 0.5799408 , 3.03953493, 1.26038972, 3.36177242,
     1.53091648, 2.53091648, 1.53091648, 1.53091648, 1.53091648],   # 0.0024805127376426625
    # sex
    [0.09174339, 0.44837065, 1.72399416, 0.90578049, 1.97168522,
     0.97168522, 2.19489482, 1.23061535, 1.23061535, 1.23061535],   # 0.008534183212646752
    [1.08144837, 1.557894  , 1.9200822 , 1.19301103, 3.34296033,
     2.58092081, 2.58092081, 1.58092081, 1.58092081, 1.58092081]    # 0.0025844802113548977
    ]
CSD_AFT_HFCR = np.array(CSD_AFT_HFCR)

iPOT_AFT_HFCR = [
    [2.07676129, 1.43765339, 0.75303359, 1.88196139, 0.97509839,
     1.15249648, 1.18074887, 1.18074887, 1.18074887, 1.18074887],   # 0.0021647235675420814
    [1.11070285, 2.66304748, 1.12255421, 3.30515274, 1.46642379,
     2.46642379, 1.46642379, 1.46642379, 1.46642379, 1.46642379],   # 0.0019019901917263354
    [0.10251991, 1.46668889, 0.81431322, 1.94521114, 0.94521114,
     2.12260923, 1.15086162, 1.15086162, 1.15086162, 1.15086162],   # 0.0032377947375094376
    [3.08494424, 2.63401198, 1.06127458, 3.24190299, 1.49631104,
     1.49631104, 1.49631104, 1.49631104, 1.49631104, 1.49631104],   # 0.006489128769567524
]
iPOT_AFT_HFCR = np.array(iPOT_AFT_HFCR)

CSD_GB_FLCHAIN = [
    # age, old, then young
    [19., 4.20198451, 5.59975194, 15.02698777, 23.72787822,
     35.65878595, 50.57745618, 75.42136566, 72.80207199, 54.98371779],   # 0.04877044157147281
    [28.04844293, 35.58575656, 40.190948, 45.10277101, 47.04230967,
     47.73855474, 46.82280427, 46.82280427, 46.82280427, 46.82280427],   # 0.0016346342173121917
    # sex, 0 and then 1
    [16.94984522, 22.41446527, 27.49781295, 28.86821742, 39.10928422,
     44.61797345, 54.18326075, 72.19986832, 75.15014928, 63.00912312],   # 0.02152650156284129
    [30.09859771, 17.37327579, 18.29288699, 31.26154136, 31.66090367,
     38.77936723, 43.2169997, 50.04430162, 44.47472699, 38.79739894]    # 0.007557202333207841
    ]
CSD_GB_FLCHAIN = np.array(CSD_GB_FLCHAIN)

iPOT_GB_FLCHAIN = [
    [44.12038544, 39.82884124, 35.1222391, 37.58157957, 34.59492591,
     36.30562427, 30.4808825, 34.38158688, 32.98924785, 31.59468724],   # 0.0009013803262434327
    [36.40051829, 37.83834409, 47.31055022, 44.18570889, 44.21081308,
     44.21081308, 44.21081308, 44.21081308, 44.21081308, 44.21081308],   # 0.00021551521268316833
    [31.32699603, 42.92668845, 41.2577547, 50.93617552, 45.42654717,
     46.61871934, 45.39803592, 48.1713728, 46.16613534, 45.77157473],   # 0.0005885821651722739
    [49.19390769, 34.74049689, 41.17503463, 30.83111294, 33.37919182,
     33.89771801, 29.29365967, 30.42102717, 31.03392559, 30.03392559],   # 0.001937147844708078
]
iPOT_GB_FLCHAIN = np.array(iPOT_GB_FLCHAIN)

CSD_CoxPH_employee = [
    # salary, high, then low
    [25.82562578, 75.96869196, 41.79887989, 56.37490806, 68.40515649,
     63.34587409, 65.57021593, 61.57021593, 59.57021593, 59.57021593],   # 0.0012634356813912703
    [26.08804741, 96.77574783, 41.92498788, 61.52165588, 83.58091664,
     65.86289636, 66.37394186, 61.27616048, 59.29782283, 59.29782283],   # 0.0006844077378557945
    ]
CSD_CoxPH_employee = np.array(CSD_CoxPH_employee)

iPOT_CoxPH_employee = [
    [58.3769396, 46.1973122, 59.93157948, 61.26775536, 59.94284371,
     59.11588809, 58.29192039, 58.29192039, 58.29192039, 58.29192039],   # 8.264507711771176e-05
    [41.40567185, 84.67082192, 61.91508044, 76.9648232, 65.26552365,
     61.1189454, 58.36741792, 57.43057187, 57.43057187, 57.43057187],   # 0.0004926018254330022
]
iPOT_CoxPH_employee = np.array(iPOT_CoxPH_employee)


CSD_MTLR_MIMIC = [
    # AGE
    [ 85.86559753, 145.66251376, 174.11567976, 205.73861592,
       221.45870247, 219.44955211, 245.73119531, 222.57727073,
       239.59535762, 246.8055148 ],  # 0.005248028507945292
    [117.4289766 , 152.07506917, 166.89040552, 190.41470504,
       195.8195185 , 202.73035266, 201.40467262, 209.75339106,
       203.24968978, 205.23321905],     # 0.0020659910060967226
    # SEX
    [ 94.87007857, 126.58542203, 157.94113384, 181.79200373,
       186.71689203, 197.87607854, 198.85977904, 195.44104875,
       192.64829757, 210.2692659 ], # 0.003539720617092762
    [108.42449556, 171.1521609 , 183.06495144, 214.36131723,
       230.56132894, 224.30382623, 248.2760889 , 236.88961304,
       250.19674982, 241.76946794], # 0.003545913299007888
    # WHITE, not white, then white
    [ 61.14349253,  88.06258375, 101.20052075, 111.92357163,
       109.57644335, 118.27429174, 112.44103447, 111.07076001,
       130.84316672, 125.46413505], # 0.0025761374740978548
    [142.1510816 , 209.67499918, 239.80556453, 284.22974933,
       307.70177762, 303.90561303, 334.69483346, 321.25990178,
       312.00188067, 326.57459879], # 0.003967611493526789
]
CSD_MTLR_MIMIC = np.array(CSD_MTLR_MIMIC)

iPOT_MTLR_MIMIC = [
    [193.11388061, 196.01158106, 182.42078576, 197.02628077,
       210.32872935, 198.1978965 , 198.20423481, 202.39184321,
       206.65238396, 222.65238396], # 0.00015969577679760406
    [176.79627669, 179.71498267, 179.37896545, 179.88847313,
       191.70628627, 185.36548265, 187.30688956, 189.92881908,
       188.45691225, 186.45691225], # 5.262805128962285e-05
    [169.64275899, 174.98341505, 167.4624998 , 165.80020093,
       185.2012781 , 181.83487485, 160.26727267, 179.69302829,
       170.05733566, 188.05733566], # 4.026009911918563e-05
    [200.26739831, 200.74314867, 194.33725141, 211.11455297,
       216.83373752, 201.7285043 , 225.24385171, 212.62763401,
       225.05196055, 221.05196055], # 0.0001822312886488403
    [108.02438671, 112.67195651, 101.59779614, 100.02661132,
       114.45908359, 100.99196634,  98.39965957, 102.69997503,
       120.0642824 , 111.0642824 ], # 5.583622903884986e-05
    [261.8857706 , 263.05460722, 260.20195507, 276.88814258,
       287.57593202, 282.57141281, 287.1114648 , 289.62068727,
       275.04501382, 298.04501382], # 0.00015542346224430034
]
iPOT_MTLR_MIMIC = np.array(iPOT_MTLR_MIMIC)

# color_list = sns.color_palette('colorblind', 2)
color_list = sns.color_palette("deep", 3)[1:]
custom_params = {"axes.spines.right": False, "axes.spines.top": False,}
sns.set_theme(style="white", rc=custom_params)

plt.rc('axes', titlesize=10)     # fontsize of the axes title
plt.rc('axes', labelsize=10)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
plt.rc('legend', fontsize=8)    # legend fontsize
plt.rc('figure', titlesize=10)  # fontsize of the figure title
height_unit = 2.4
width_unit = 2.8
plt.rcParams['font.size'] = 12


# d-cal
default = np.linspace(0, 1, 11)

dcal_csd = CSD_AFT_HFCR[0]
dcal_csd = dcal_csd.cumsum() / dcal_csd.sum()
dcal_csd = np.insert(dcal_csd, 0, 0)
dcal_ipot = iPOT_AFT_HFCR[0]
dcal_ipot = dcal_ipot.cumsum() / dcal_ipot.sum()
dcal_ipot = np.insert(dcal_ipot, 0, 0)
plt.figure(figsize=(width_unit, height_unit))
plt.plot([0, 1], [0, 1], ls='dashed', c='grey', label='Perfect Cal.')
plt.plot(default, dcal_csd, label='CSD', marker='o', markersize=9, c=color_list[0], linewidth=2,
          clip_on=False, zorder=10)
plt.plot(default, dcal_ipot, label='CSD-iPOT', marker='o', markersize=9, c=color_list[1], linewidth=2,
            clip_on=False, zorder=10)
plt.ylim([-0.0, 1.0])
plt.xlim([-0.0, 1.0])
plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
plt.xlabel('Predicted Survival Prob.')
plt.ylabel('Observed Survival Prob.')
plt.grid(False)
# plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig(f'figs/case_study/AFT_HFCR_dcal.png', dpi=400)
plt.close()

dcal_csd = CSD_GB_FLCHAIN[2]
dcal_csd = dcal_csd.cumsum() / dcal_csd.sum()
dcal_csd = np.insert(dcal_csd, 0, 0)
dcal_ipot = iPOT_GB_FLCHAIN[2]
dcal_ipot = dcal_ipot.cumsum() / dcal_ipot.sum()
dcal_ipot = np.insert(dcal_ipot, 0, 0)
plt.figure(figsize=(width_unit, height_unit))
plt.plot([0, 1], [0, 1], ls='dashed', c='grey', label='Optimal')
plt.plot(default, dcal_csd, label='CSD', marker='o', markersize=9, c=color_list[0], linewidth=2,
          clip_on=False, zorder=10)
plt.plot(default, dcal_ipot, label='CSD-iPOT', marker='o', markersize=9, c=color_list[1], linewidth=2,
            clip_on=False, zorder=10)
plt.ylim([-0.0, 1.0])
plt.xlim([-0.0, 1.0])
plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
plt.xlabel('Predicted Survival Prob.')
plt.ylabel('Observed Survival Prob.')
plt.grid(False)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig(f'figs/case_study/GB_FLCHAIN_dcal.png', dpi=400)
plt.close()

dcal_csd = CSD_CoxPH_employee[0]
dcal_csd = dcal_csd.cumsum() / dcal_csd.sum()
dcal_csd = np.insert(dcal_csd, 0, 0)
dcal_ipot = iPOT_CoxPH_employee[0]
dcal_ipot = dcal_ipot.cumsum() / dcal_ipot.sum()
dcal_ipot = np.insert(dcal_ipot, 0, 0)
plt.figure(figsize=(width_unit, height_unit))
plt.plot([0, 1], [0, 1], ls='dashed', c='grey', label='Perfect Cal.')
plt.plot(default, dcal_csd, label='CSD', marker='o', markersize=9, c=color_list[0], linewidth=2,
          clip_on=False, zorder=10)
plt.plot(default, dcal_ipot, label='CSD-iPOT', marker='o', markersize=9, c=color_list[1], linewidth=2,
            clip_on=False, zorder=10)
plt.ylim([-0.0, 1.0])
plt.xlim([-0.0, 1.0])
plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
plt.xlabel('Predicted Survival Prob.')
plt.ylabel('Observed Survival Prob.')
plt.grid(False)
# plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig(f'figs/case_study/CoxPH_employee_dcal.png', dpi=400)

dcal_csd = CSD_MTLR_MIMIC[4]
dcal_csd = dcal_csd.cumsum() / dcal_csd.sum()
dcal_csd = np.insert(dcal_csd, 0, 0)
dcal_ipot = iPOT_MTLR_MIMIC[4]
dcal_ipot = dcal_ipot.cumsum() / dcal_ipot.sum()
dcal_ipot = np.insert(dcal_ipot, 0, 0)
plt.figure(figsize=(width_unit, height_unit))
plt.plot([0, 1], [0, 1], ls='dashed', c='grey', label='Perfect Cal.')
plt.plot(default, dcal_csd, label='CSD', marker='o', markersize=9, c=color_list[0], linewidth=2,
          clip_on=False, zorder=10)
plt.plot(default, dcal_ipot, label='CSD-iPOT', marker='o', markersize=9, c=color_list[1], linewidth=2,
            clip_on=False, zorder=10)
plt.ylim([-0.0, 1.0])
plt.xlim([-0.0, 1.0])
plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
plt.xlabel('Predicted Survival Prob.')
plt.ylabel('Observed Survival Prob.')
plt.grid(False)
# plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig(f'figs/case_study/MTLR_MIMIC_dcal.png', dpi=400)
plt.close()

# km-cal

file_path = f'results/calibration/AFT_HFCR/'
quan_preds_csd = np.load(file_path + 'quan_preds_csd.npy')
quan_preds_ipot = np.load(file_path + 'quan_preds_ipot.npy')
quan_levels_csd = np.load(file_path + 'quan_levels_csd.npy')
quan_levels_ipot = np.load(file_path + 'quan_levels_ipot.npy')
t_test_csd = np.load(file_path + 't_test_csd.npy')
t_test_ipot = np.load(file_path + 't_test_ipot.npy')
e_test_csd = np.load(file_path + 'e_test_csd.npy')
e_test_ipot = np.load(file_path + 'e_test_ipot.npy')
x_test_csd = np.load(file_path + 'x_test_csd.npy')
x_test_ipot = np.load(file_path + 'x_test_ipot.npy')

# make sure the t_test, e_test, x_test are the same
assert np.allclose(t_test_csd, t_test_ipot)
assert np.allclose(e_test_csd, e_test_ipot)
assert np.allclose(x_test_csd, x_test_ipot)

# old age means the first feature is > 0
cond = (x_test_csd[:, 0] > 0)
kmf = KaplanMeier(t_test_csd[cond], e_test_csd[cond])
km_times = kmf.survival_times
km_probs = kmf.survival_probabilities

survival_curves_csd = quantile_to_survival(quan_levels_csd, quan_preds_csd[cond], km_times, interpolate='Pchip')
mean_curve_csd = np.mean(survival_curves_csd, axis=0)
survival_curves_ipot = quantile_to_survival(quan_levels_ipot, quan_preds_ipot[cond], km_times, interpolate='Pchip')
mean_curve_ipot = np.mean(survival_curves_ipot, axis=0)

plt.figure(figsize=(width_unit, height_unit))
plt.step(km_times, km_probs, label='Perfect Cal.', c='grey', linestyle='dashed')
plt.step(km_times, mean_curve_csd, label='Non-CSD', c=color_list[0], linewidth=2,
         clip_on=False, zorder=10)
plt.step(km_times, mean_curve_ipot, label='CSD', c=color_list[1], linewidth=2,
         clip_on=False, zorder=10)
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
# plt.xlim([0, km_times.max()])
plt.ylim([0, 1])
# ax.legend()
plt.savefig(f'figs/case_study/AFT_HFCR_kmcal.png', dpi=400, bbox_inches='tight')
plt.close()


file_path = f'results/calibration/GB_FLCHAIN/'
quan_preds_csd = np.load(file_path + 'quan_preds_csd.npy')
quan_preds_ipot = np.load(file_path + 'quan_preds_ipot.npy')
quan_levels_csd = np.load(file_path + 'quan_levels_csd.npy')
quan_levels_ipot = np.load(file_path + 'quan_levels_ipot.npy')
t_test_csd = np.load(file_path + 't_test_csd.npy')
t_test_ipot = np.load(file_path + 't_test_ipot.npy')
e_test_csd = np.load(file_path + 'e_test_csd.npy')
e_test_ipot = np.load(file_path + 'e_test_ipot.npy')
x_test_csd = np.load(file_path + 'x_test_csd.npy')
x_test_ipot = np.load(file_path + 'x_test_ipot.npy')

# make sure the t_test, e_test, x_test are the same
assert np.allclose(t_test_csd, t_test_ipot)
assert np.allclose(e_test_csd, e_test_ipot)
assert np.allclose(x_test_csd, x_test_ipot)

# old age means the second feature is = 0
cond = (x_test_csd[:, 1] == 0)
kmf = KaplanMeier(t_test_csd[cond], e_test_csd[cond])
km_times = kmf.survival_times
km_probs = kmf.survival_probabilities

survival_curves_csd = quantile_to_survival(quan_levels_csd, quan_preds_csd[cond], km_times, interpolate='Pchip')
mean_curve_csd = np.mean(survival_curves_csd, axis=0)
survival_curves_ipot = quantile_to_survival(quan_levels_ipot, quan_preds_ipot[cond], km_times, interpolate='Pchip')
mean_curve_ipot = np.mean(survival_curves_ipot, axis=0)

plt.figure(figsize=(width_unit, height_unit))
plt.step(km_times, km_probs, label='Perfect Cal.', c='grey', linestyle='dashed')
plt.step(km_times, mean_curve_csd, label='Non-CSD', c=color_list[0], linewidth=2,
         clip_on=False, zorder=10)
plt.step(km_times, mean_curve_ipot, label='CSD', c=color_list[1], linewidth=2,
         clip_on=False, zorder=10)
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
# plt.xlim([0, km_times.max()])
plt.ylim([0, 1])
# ax.legend()
plt.savefig(f'figs/case_study/GB_FLCHAIN_kmcal.png', dpi=400, bbox_inches='tight')
plt.close()


file_path = f'results/calibration/CoxPH_employee/'
quan_preds_csd = np.load(file_path + 'quan_preds_csd.npy')
quan_preds_ipot = np.load(file_path + 'quan_preds_ipot.npy')
quan_levels_csd = np.load(file_path + 'quan_levels_csd.npy')
quan_levels_ipot = np.load(file_path + 'quan_levels_ipot.npy')
t_test_csd = np.load(file_path + 't_test_csd.npy')
t_test_ipot = np.load(file_path + 't_test_ipot.npy')
e_test_csd = np.load(file_path + 'e_test_csd.npy')
e_test_ipot = np.load(file_path + 'e_test_ipot.npy')
x_test_csd = np.load(file_path + 'x_test_csd.npy')
x_test_ipot = np.load(file_path + 'x_test_ipot.npy')

# make sure the t_test, e_test, x_test are the same
assert np.allclose(t_test_csd, t_test_ipot)
assert np.allclose(e_test_csd, e_test_ipot)
assert np.allclose(x_test_csd, x_test_ipot)

# old age means the 7th feature is > 0
cond = (x_test_csd[:, 6] > 0)
kmf = KaplanMeier(t_test_csd[cond], e_test_csd[cond])
km_times = kmf.survival_times
km_probs = kmf.survival_probabilities

survival_curves_csd = quantile_to_survival(quan_levels_csd, quan_preds_csd[cond], km_times, interpolate='Pchip')
mean_curve_csd = np.mean(survival_curves_csd, axis=0)
survival_curves_ipot = quantile_to_survival(quan_levels_ipot, quan_preds_ipot[cond], km_times, interpolate='Pchip')
mean_curve_ipot = np.mean(survival_curves_ipot, axis=0)

plt.figure(figsize=(width_unit, height_unit))
plt.step(km_times, km_probs, label='Perfect Cal.', c='grey', linestyle='dashed')
plt.step(km_times, mean_curve_csd, label='CSD', c=color_list[0], linewidth=2,
         clip_on=False, zorder=10)
plt.step(km_times, mean_curve_ipot, label='CSD-iPOT', c=color_list[1], linewidth=2,
         clip_on=False, zorder=10)
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
# plt.xlim([0, km_times.max()])
plt.ylim([0, 1])
# ax.legend()
plt.savefig(f'figs/case_study/CoxPH_employee_kmcal.png', dpi=400, bbox_inches='tight')
plt.close()


file_path = f'results/calibration/MTLR_MIMIC/'
quan_preds_csd = np.load(file_path + 'quan_preds_csd.npy')
quan_preds_ipot = np.load(file_path + 'quan_preds_ipod.npy')
quan_levels_csd = np.load(file_path + 'quan_levels_csd.npy')
quan_levels_ipot = np.load(file_path + 'quan_levels_ipod.npy')
t_test_csd = np.load(file_path + 't_test_csd.npy')
t_test_ipot = np.load(file_path + 't_test_ipod.npy')
e_test_csd = np.load(file_path + 'e_test_csd.npy')
e_test_ipot = np.load(file_path + 'e_test_ipod.npy')
x_test_csd = np.load(file_path + 'x_test_csd.npy')
x_test_ipot = np.load(file_path + 'x_test_ipod.npy')

# make sure the t_test, e_test, x_test are the same
assert np.allclose(t_test_csd, t_test_ipot)
assert np.allclose(e_test_csd, e_test_ipot)
assert np.allclose(x_test_csd, x_test_ipot)

# non-white means the 50th feature is == 0
cond = (x_test_csd[:, 50] == 0)
kmf = KaplanMeier(t_test_csd[cond], e_test_csd[cond])
km_times = kmf.survival_times
km_probs = kmf.survival_probabilities

survival_curves_csd = quantile_to_survival(quan_levels_csd, quan_preds_csd[cond], km_times, interpolate='Pchip')
mean_curve_csd = np.mean(survival_curves_csd, axis=0)
survival_curves_ipot = quantile_to_survival(quan_levels_ipot, quan_preds_ipot[cond], km_times, interpolate='Pchip')
mean_curve_ipot = np.mean(survival_curves_ipot, axis=0)

plt.figure(figsize=(width_unit, height_unit))
plt.step(km_times, km_probs, label='Perfect Cal.', c='grey', linestyle='dashed')
plt.step(km_times, mean_curve_csd, label='CSD', c=color_list[0], linewidth=2,
         clip_on=False, zorder=10)
plt.step(km_times, mean_curve_ipot, label='CSD-iPOT', c=color_list[1], linewidth=2,
         clip_on=False, zorder=10)
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
# plt.xlim([0, km_times.max()])
plt.ylim([0, 1])
# ax.legend()
plt.savefig(f'figs/case_study/MTLR_MIMIC_kmcal.png', dpi=400, bbox_inches='tight')
plt.close()
