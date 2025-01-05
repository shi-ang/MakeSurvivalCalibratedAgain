import numpy as np
import pandas as pd
import h5py
from collections import defaultdict
from sksurv.datasets.base import _get_data_path
from sksurv.io import loadarff
from ucimlrepo import fetch_ucirepo
import scipy


def make_survival_data(
        dataset: str
) -> tuple[pd.DataFrame, list[str]]:
    if dataset == "SUPPORT":
        return make_support()
    elif dataset == "METABRIC":
        return make_metabric()
    elif dataset == "NACD":
        return make_nacd()
    elif dataset == "DLBCL":
        return make_dlbcl()
    elif dataset == "FLCHAIN":
        return make_flchain()
    elif dataset == "GBSG":
        return make_gbsg()
    elif dataset == "WHAS500":
        return make_whas500()
    elif dataset == "PBC":
        return make_pbc()
    elif dataset == "VALCT":
        return make_valct()
    elif dataset == "GBM":
        return make_gbm()
    elif dataset == "HFCR":
        return make_heart_failure()
    elif dataset == "churn":
        return make_churn()
    elif dataset == "employee":
        return make_employee_retention()
    elif dataset == "PDM":
        return make_pdm()
    elif dataset == "MIMIC-IV_all":
        return make_mimic_iv_all()
    elif dataset == "SEER_brain":
        return make_seer_brain()
    elif dataset == "SEER_stomach":
        return make_seer_stomach()
    elif dataset == "SEER_liver":
        return make_seer_liver()
    else:
        raise ValueError("Dataset name not recognized.")


def make_support() -> tuple[pd.DataFrame, list[str]]:
    """Downloads and preprocesses the SUPPORT dataset from [1]_.

    The missing values are filled using either the recommended
    standard values, the mean (for continuous variables) or the mode
    (for categorical variables).
    Refer to the dataset description at
    https://hbiostat.org/data/repo/supportdesc for more information.

    Returns
    -------
    pd.DataFrame
        Processed covariates for one patient in each row.
    list[str]
        List of columns to standardize.

    References
    ----------
    [1] W. A. Knaus et al., The SUPPORT Prognostic Model: Objective Estimates of Survival
    for Seriously Ill Hospitalized Adults, Ann Intern Med, vol. 122, no. 3, p. 191, Feb. 1995.
    """
    url = "https://hbiostat.org/data/repo/support2csv.zip"

    # Remove other target columns and other model predictions
    cols_to_drop = ["hospdead", "slos", "charges", "totcst", "totmcst", "avtisst", "sfdm2",
                    "adlp", "adls", "dzgroup",  # "adlp", "adls", and "dzgroup" were used in other preprocessing steps,
                    # see https://github.com/autonlab/auton-survival/blob/master/auton_survival/datasets.py
                    "sps", "aps", "surv2m", "surv6m", "prg2m", "prg6m", "dnr", "dnrday", "hday"]

    # `death` is the overall survival event indicator
    # `d.time` is the time to death from any cause or censoring
    data = (pd.read_csv(url)
            .drop(cols_to_drop, axis=1)
            .rename(columns={"d.time": "time", "death": "event"}))
    data["event"] = data["event"].astype(int)

    data["ca"] = (data["ca"] == "metastatic").astype(int)

    # use recommended default values from official dataset description ()
    # or mean (for continuous variables)/mode (for categorical variables) if not given
    fill_vals = {
        "alb": 3.5,
        "pafi": 333.3,
        "bili": 1.01,
        "crea": 1.01,
        "bun": 6.51,
        "wblc": 9,
        "urine": 2502,
        "edu": data["edu"].mean(),
        "ph": data["ph"].mean(),
        "glucose": data["glucose"].mean(),
        "scoma": data["scoma"].mean(),
        "meanbp": data["meanbp"].mean(),
        "hrt": data["hrt"].mean(),
        "resp": data["resp"].mean(),
        "temp": data["temp"].mean(),
        "sod": data["sod"].mean(),
        "income": data["income"].mode()[0],
        "race": data["race"].mode()[0],
    }
    data = data.fillna(fill_vals)

    with pd.option_context("future.no_silent_downcasting", True):
        data.sex = data.sex.replace({'male': 1, 'female': 0}).infer_objects(copy=False)
        data.income = data.income.replace(
            {'under $11k': 0, '$11-$25k': 1, '$25-$50k': 2, '>$50k': 3}).infer_objects(copy=False)
    skip_cols = ['event', 'sex', 'time', 'dzclass', 'race', 'diabetes', 'dementia', 'ca']
    cols_standardize = list(set(data.columns.to_list()).symmetric_difference(skip_cols))

    # one-hot encode categorical variables
    onehot_cols = ["dzclass", "race"]
    data = pd.get_dummies(data, columns=onehot_cols, drop_first=True)
    data = data.rename(columns={"dzclass_COPD/CHF/Cirrhosis": "dzclass_COPD"})

    data.reset_index(drop=True, inplace=True)
    return data, cols_standardize


def make_nacd() -> tuple[pd.DataFrame, list[str]]:
    cols_to_drop = ['PERFORMANCE_STATUS', 'STAGE_NUMERICAL', 'AGE65']
    data = pd.read_csv("data/NACD_Full.csv").drop(cols_to_drop, axis=1).rename(columns={"delta": "event"})

    data = data.drop(data[data["time"] <= 0].index)  # remove patients with negative or zero survival time
    data.reset_index(drop=True, inplace=True)
    cols_standardize = ['BOX1_SCORE', 'BOX2_SCORE', 'BOX3_SCORE', 'BMI', 'WEIGHT_CHANGEPOINT',
                        'AGE', 'GRANULOCYTES', 'LDH_SERUM', 'LYMPHOCYTES',
                        'PLATELET', 'WBC_COUNT', 'CALCIUM_SERUM', 'HGB', 'CREATININE_SERUM', 'ALBUMIN']
    return data, cols_standardize


def make_metabric() -> tuple[pd.DataFrame, list[str]]:
    data = pd.read_csv("data/Metabric.csv").rename(columns={"delta": "event"})
    cols_standardize = ['age_at_diagnosis', 'size', 'lymph_nodes_positive', 'stage', 'lymph_nodes_removed', 'NPI']
    return data, cols_standardize


def make_dlbcl() -> tuple[pd.DataFrame, list[str]]:
    data = pd.read_csv("data/DLBCL.csv").rename(columns={"delta": "event"})
    assert not data.isnull().values.any(), "Dataset contains NaNs"
    skip_cols = ['event', 'time']
    cols_standardize = list(set(data.columns.to_list()).symmetric_difference(skip_cols))
    return data, cols_standardize


def make_flchain() -> tuple[pd.DataFrame, list[str]]:
    # flchain dataset: relationship between serum free light chain (FLC) and mortality
    # see: https://vincentarelbundock.github.io/Rdatasets/doc/survival/flchain.html
    cols_to_drop = ["chapter"]  # only dead patients has chapter information
    data = pd.read_csv("data/flchain.csv").drop(cols_to_drop, axis=1).rename(columns={"futime": "time",
                                                                                      "death": "event"})
    data = data.drop(data[data["time"] <= 0].index)  # remove patients with negative or zero survival time
    data.reset_index(drop=True, inplace=True)
    with pd.option_context("future.no_silent_downcasting", True):
        data.sex = data.sex.replace({'M': 1, 'F': 0}).infer_objects(copy=False)
    # processing see: https://github.com/paidamoyo/adversarial_time_to_event/blob/master/data/flchain/flchain_data.py
    data = data.fillna({"creatinine": data["creatinine"].median()})
    onehot_cols = ["sample.yr", "flc.grp"]
    data = pd.get_dummies(data, columns=onehot_cols, drop_first=True)
    skip_cols = {'event', 'time', 'sex', 'mgus'}
    assert not data.isnull().values.any(), "Dataset contains NaNs"
    cols_standardize = list(set(data.columns.to_list()).symmetric_difference(skip_cols))
    return data, cols_standardize


def make_mimic_iv_all() -> tuple[pd.DataFrame, list[str]]:
    data = pd.read_csv("data/MIMIC_IV_all_cause_failure.csv")
    skip_cols = ['event', 'is_male', 'time', 'is_white', 'renal', 'cns', 'coagulation', 'cardiovascular']
    cols_standardize = list(set(data.columns.to_list()).symmetric_difference(skip_cols))
    return data, cols_standardize


def make_gbsg_old() -> tuple[pd.DataFrame, list[str]]:
    """
    Rotterdam & German Breast Cancer Study Group (GBSG)

    A combination of the Rotterdam tumor bank and the German Breast Cancer Study Group.

    This is the processed data set used in the DeepSurv paper (Katzman et al. 2018), and details
    can be found at https://doi.org/10.1186/s12874-018-0482-1

    The original data file 'gbsg_cancer_train_test.h5' is downloaded from
    https://github.com/jaredleekatzman/DeepSurv/blob/master/experiments/data/gbsg/gbsg_cancer_train_test.h5
    """
    data = defaultdict(dict)
    with h5py.File('data/gbsg_cancer_train_test.h5') as f:
        for ds in f:
            for array in f[ds]:
                data[ds][array] = f[ds][array][:]
    train = _make_df(data['train'])
    test = _make_df(data['test'])
    df = pd.concat([train, test]).reset_index(drop=True).rename(columns={"duration": "time"})
    cols_standardize = ['x3', 'x4', 'x5', 'x6']

    del data, train, test
    return df, cols_standardize


def make_gbsg() -> tuple[pd.DataFrame, list[str]]:
    """
    German Breast Cancer Study Group (GBSG)

    This dataset is downloaded from `survival` package in R.
    The data description can be found at https://rdrr.io/cran/survival/man/gbsg.html
    """
    cols_to_drop = ['pid']
    data = pd.read_csv("data/GBSG.csv").drop(cols_to_drop, axis=1).rename(
        columns={"status": "event", "rfstime": "time"})

    cols_standardize = ['age', 'size', 'grade', 'nodes', 'pgr', 'er']
    return data, cols_standardize


def _make_df(data):
    x = data['x']
    t = data['t']
    d = data['e']

    colnames = ['x'+str(i) for i in range(x.shape[1])]
    df = (pd.DataFrame(x, columns=colnames)
          .assign(duration=t)
          .assign(event=d))
    return df


def make_whas500() -> tuple[pd.DataFrame, list[str]]:
    """
    Worcester Heart Attack Study dataset. See [1] and [2] for details.
    [1] https://web.archive.org/web/20170114043458/http://www.umass.edu/statdata/statdata/data/
    [2] Hosmer, D., Lemeshow, S., May, S.: “Applied Survival Analysis: Regression Modeling of Time
    to Event Data.” John Wiley & Sons, Inc. (2008)
    """
    fn = _get_data_path("whas500.arff")
    data = loadarff(fn).rename(columns={"fstat": "event", "lenfol": "time"})
    data = data.astype(float)
    cols_standardize = ['age', 'bmi', 'diasbp', 'hr', 'los', 'sysbp']
    return data, cols_standardize


def make_valct() -> tuple[pd.DataFrame, list[str]]:
    """
    Veterans’ Administration Lung Cancer Trial

    [1] Kalbfleisch, J.D., Prentice, R.L.: “The Statistical Analysis of Failure Time Data.”
    John Wiley & Sons, Inc. (2002)
    """
    fn = _get_data_path("veteran.arff")
    data = loadarff(fn).rename(columns={"Status": "event", "Survival_in_days": "time"})
    with pd.option_context("future.no_silent_downcasting", True):
        data.event = data.event.replace({'dead': 1, 'censored': 0}).infer_objects(copy=False)
        data.Prior_therapy = data.Prior_therapy.replace({'no': 0, 'yes': 1}).infer_objects(copy=False)
        data.Treatment = data.Treatment.replace({'standard': 0, 'test': 1}).infer_objects(copy=False)
    data = pd.get_dummies(data, columns=['Celltype'], drop_first=True)
    data = data.astype(float)

    cols_standardize = ['Age_in_years', 'Karnofsky_score', 'Months_from_Diagnosis']

    return data, cols_standardize


def make_gbm() -> tuple[pd.DataFrame, list[str]]:
    data = pd.read_csv("data/GBM.clin.merged.picked.csv").rename(columns={"delta": "event"})
    data.drop(columns=["Composite Element REF", "tumor_tissue_site"], inplace=True)  # Columns with only one value
    data = data[data.time.notna()]  # Unknown censor/event time
    data = data.drop(data[data["time"] <= 0].index)  # remove patients with negative or zero survival time
    data.reset_index(drop=True, inplace=True)

    # Preprocess and fill missing values
    with pd.option_context("future.no_silent_downcasting", True):
        data.gender = data.gender.replace({'male': 1, 'female': 0}).infer_objects(copy=False)
        data.radiation_therapy = data.radiation_therapy.replace({'yes': 1, 'no': 0}).infer_objects(copy=False)
        data.ethnicity = data.ethnicity.replace(
            {'not hispanic or latino': 0, 'hispanic or latino': 1}).infer_objects(copy=False)
    # one-hot encode categorical variables
    onehot_cols = ["histological_type", "race"]
    data = pd.get_dummies(data, columns=onehot_cols, drop_first=True)
    fill_vals = {
        "radiation_therapy": data["radiation_therapy"].median(),
        "karnofsky_performance_score": data["karnofsky_performance_score"].median(),
        "ethnicity": data["ethnicity"].median()
    }
    data = data.fillna(fill_vals)
    data.columns = data.columns.str.replace(" ", "_")

    cols_standardize = ['years_to_birth', 'date_of_initial_pathologic_diagnosis', 'karnofsky_performance_score']
    return data, cols_standardize


def make_seer_liver() -> tuple[pd.DataFrame, list[str]]:
    """
    Preprocess the SEER liver cancer dataset.
    """
    data = pd.read_csv("data/SEER/Liver.csv").rename(columns={"Survival months": "time"})
    data = data.drop(data[data["time"] <= 0].index)  # remove patients with negative or zero survival time
    data.reset_index(drop=True, inplace=True)

    skip_cols = ['event', 'time', 'Sex']
    cols_standardize = list(set(data.columns.to_list()).symmetric_difference(skip_cols))

    return data, cols_standardize


def make_seer_brain() -> tuple[pd.DataFrame, list[str]]:
    """
    Preprocess the SEER brain cancer dataset.
    """
    data = pd.read_csv("data/SEER/Brain.csv").rename(columns={"Survival months": "time"})
    data = data.drop(data[data["time"] <= 0].index)  # remove patients with negative or zero survival time
    data.reset_index(drop=True, inplace=True)

    skip_cols = ['event', 'time', 'Sex', 'Behavior recode for analysis',
                 'SEER historic stage A (1973-2015)', 'RX Summ--Scope Reg LN Sur (2003+)']
    cols_standardize = list(set(data.columns.to_list()).symmetric_difference(skip_cols))

    return data, cols_standardize


def make_seer_stomach() -> tuple[pd.DataFrame, list[str]]:
    """
    Preprocess the SEER stomach cancer dataset.
    """
    data = pd.read_csv("data/SEER/Stomach.csv").rename(columns={"Survival months": "time"})
    data = data.drop(data[data["time"] <= 0].index)  # remove patients with negative or zero survival time
    data.reset_index(drop=True, inplace=True)

    skip_cols = ['event', 'time', 'Sex']
    cols_standardize = list(set(data.columns.to_list()).symmetric_difference(skip_cols))

    return data, cols_standardize


def make_pbc():
    """
    Preprocess the Cirrhosis Patient Survival Prediction dataset.

    Link: https://archive.ics.uci.edu/dataset/878/cirrhosis+patient+survival+prediction+dataset-1
    Paper: https://pubmed.ncbi.nlm.nih.gov/2737595/
    """
    cirrhosis = fetch_ucirepo(id=878)

    cols_to_drop = ['ID']
    data = cirrhosis.data.original.drop(cols_to_drop, axis=1).rename(columns={"Status": "event",
                                                                              "N_Days": "time"})
    with pd.option_context("future.no_silent_downcasting", True):
        data = data.replace({'NaNN': np.nan}).infer_objects(copy=False)
        data.event = data.event.replace({'C': 0, 'CL': 0, 'D': 1}).infer_objects(copy=False)
        data.Drug = data.Drug.replace({'D-penicillamine': 0, 'Placebo': 1}).infer_objects(copy=False)
        data.Sex = data.Sex.replace({'M': 1, 'F': 0}).infer_objects(copy=False)
        data.Ascites = data.Ascites.replace({'N': 0, 'Y': 1}).infer_objects(copy=False)
        data.Hepatomegaly = data.Hepatomegaly.replace({'N': 0, 'Y': 1}).infer_objects(copy=False)
        data.Spiders = data.Spiders.replace({'N': 0, 'Y': 1}).infer_objects(copy=False)
        data.Edema = data.Edema.replace({'N': 0, 'Y': 1, 'S': 0.5}).infer_objects(copy=False)
    data.Cholesterol = pd.to_numeric(data.Cholesterol, errors='coerce')
    data.Copper = pd.to_numeric(data.Copper, errors='coerce')
    data.Tryglicerides = pd.to_numeric(data.Tryglicerides, errors='coerce')
    data.Platelets = pd.to_numeric(data.Platelets, errors='coerce')

    fill_vals = {
        "Drug": data.Drug.mode()[0],
        "Ascites": data.Ascites.mode()[0],
        "Hepatomegaly": data.Hepatomegaly.mode()[0],
        "Spiders": data.Spiders.mode()[0],
        "Cholesterol": data.Cholesterol.mean(),
        "Copper": data.Copper.mean(),
        "Alk_Phos": data.Alk_Phos.mean(),
        "SGOT": data.SGOT.mean(),
        "Tryglicerides": data.Tryglicerides.mean(),
        "Platelets": data.Platelets.mean(),
        "Prothrombin": data.Prothrombin.mean(),
        "Stage": data.Stage.mode()[0],
    }

    data = data.fillna(fill_vals)
    data.reset_index(drop=True, inplace=True)

    skip_cols = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema', 'Stage', 'event', 'time']
    cols_standardize = list(set(data.columns.to_list()).symmetric_difference(skip_cols))
    return data, cols_standardize


def make_heart_failure():
    """
    Preprocess the Heart Failure Prediction dataset.

    Link: https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records
    Paper: https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5
    """
    heart_failure = fetch_ucirepo(id=519)

    X = heart_failure.data.features
    y = heart_failure.data.targets
    data = pd.concat([X, y], axis=1).rename(columns={"death_event": "event"})

    cols_standardize = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets',
                        'serum_creatinine', 'serum_sodium']

    return data, cols_standardize


def make_churn():
    """
    Predicting when your customers will churn.

    Data description: https://square.github.io/pysurvival/tutorials/churn.html
    Data downloaded from PySurvival: https://github.com/square/pysurvival/tree/master/pysurvival/datasets

    Link: https://www.kaggle.com/blastchar/telco-customer-churn
    """
    churn = pd.read_csv("data/churn.csv").rename(columns={"months_active": "time", "churned": "event"})
    churn.event = churn.event.astype(int)

    with pd.option_context("future.no_silent_downcasting", True):
        churn.product_travel_expense = churn.product_travel_expense.replace({
            'No': 0,
            'Free-Trial': 1,
            'Active': 2}
        ).infer_objects(copy=False)
        churn.product_payroll = churn.product_payroll.replace({
            'No': 0,
            'Free-Trial': 1,
            'Active': 2}
        ).infer_objects(copy=False)
        churn.product_accounting = churn.product_accounting.replace({
            'No': 0,
            'Free-Trial': 1,
            'Active': 2}
        ).infer_objects(copy=False)
        churn.company_size = churn.company_size.replace({
            'self-employed': 0,
            '1-10': 1,
            '10-50': 2,
            '50-100': 3,
            '100-250': 4, }
        ).infer_objects(copy=False)
    # creating one-hot vectors for categorical variables
    cat_cols = ['us_region']
    data = pd.get_dummies(churn, columns=cat_cols, drop_first=True)

    data = data.drop(data[data["time"] <= 0].index)  # remove patients with negative or zero survival time
    data.reset_index(drop=True, inplace=True)

    cols_standardize = ['product_data_storage', 'product_travel_expense', 'product_payroll', 'product_accounting',
                        'csat_score', 'articles_viewed', 'marketing_emails_clicked',
                        'minutes_customer_support', 'company_size']

    return data, cols_standardize


def make_employee_retention():
    """
    Predicting employee retention.

    Data description:
        https://square.github.io/pysurvival/tutorials/employee_retention.html
    Data downloaded from PySurvival:
        https://github.com/square/pysurvival/blob/master/pysurvival/datasets/employee_attrition.csv
    """
    retention = pd.read_csv("data/employee_attrition.csv").rename(
        columns={"time_spend_company": "time", "left": "event"})

    with pd.option_context("future.no_silent_downcasting", True):
        retention.salary = retention.salary.replace({
            'low': 0,
            'medium': 1,
            'high': 2}
        ).infer_objects(copy=False)
    # creating one-hot vectors for categorical variables
    cat_cols = ['department']
    data = pd.get_dummies(retention, columns=cat_cols, drop_first=True)

    data = data.drop_duplicates(keep='first').reset_index(drop=True)

    cols_standardize = ['satisfaction_level', 'last_evaluation', 'number_projects', 'average_montly_hours',
                        'work_accident', 'promotion_last_5years', 'salary']

    return data, cols_standardize


def make_pdm():
    """
    Predictive maintenance dataset.

    Data description:
        https://square.github.io/pysurvival/tutorials/maintenance.html
    Data downloaded from PySurvival:
        https://github.com/square/pysurvival/blob/master/pysurvival/datasets/maintenance.csv
    """
    pdm = pd.read_csv("data/maintenance.csv", sep=";").rename(
        columns={"lifetime": "time", "broken": "event"})

    cat_cols = ['team', 'provider']
    data = pd.get_dummies(pdm, columns=cat_cols, drop_first=True)

    cols_standardize = ['pressureInd', 'moistureInd', 'temperatureInd',]

    return data, cols_standardize
