import os
import numpy as np
import pandas as pd
import pickle
from sksurv.datasets.base import _get_data_path
from sksurv.io import loadarff


# current file path
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

def make_survival_data(
        dataset: str
) -> pd.DataFrame:
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
        return make_gbsg2()
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


def add_prefix(
    df: pd.DataFrame,
    con_list: list[str],
) -> pd.DataFrame:
    """
    Add a prefix ("num_") to the continuous columns in the DataFrame.
    """
    df = df.copy()
    for col in con_list:
        if col in df.columns:
            df.rename(columns={col: f"num_{col}"}, inplace=True)
    return df


def make_support() -> pd.DataFrame:
    """Downloads and preprocesses the SUPPORT dataset from [1]_.

    The missing values are filled using either the recommended
    standard values, the mean (for continuous variables) or the mode
    (for categorical variables).
    Refer to the dataset description at
    https://hbiostat.org/data/repo/supportdesc for more information.
    Download from https://hbiostat.org/data/repo/support2csv.zip
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
    # url = "https://hbiostat.org/data/repo/support2csv.zip"

    # Remove other target columns and other model predictions
    cols_to_drop = [
        "hospdead",
        "slos",
        "charges",
        "totcst",
        "totmcst",
        "avtisst",
        "sfdm2",
        "adlp",
        "adls",
        "dzgroup",  # "adlp", "adls", and "dzgroup" were used in other preprocessing steps,
        # see https://github.com/autonlab/auton-survival/blob/master/auton_survival/datasets.py
        "sps",
        "aps",
        "surv2m",
        "surv6m",
        "prg2m",
        "prg6m",
        "dnr",
        "dnrday",
        "hday",
    ]

    # `death` is the overall survival event indicator
    # `d.time` is the time to death from any cause or censoring
    # df = pd.read_csv(url).drop(cols_to_drop, axis=1).rename(columns={"d.time": "time", "death": "event"})
    df = pd.read_csv(f"{CURRENT_PATH}/support2.csv").drop(cols_to_drop, axis=1).rename(columns={"d.time": "time", "death": "event"})
    df["event"] = df["event"].astype(int)

    df["ca"] = (df["ca"] == "metastatic").astype(int)

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
        # "edu": df["edu"].mean(),
        # "ph": df["ph"].mean(),
        # "glucose": df["glucose"].mean(),
        # "scoma": df["scoma"].mean(),
        # "meanbp": df["meanbp"].mean(),
        # "hrt": df["hrt"].mean(),
        # "resp": df["resp"].mean(),
        # "temp": df["temp"].mean(),
        # "sod": df["sod"].mean(),
        # "income": df["income"].mode()[0],
        # "race": df["race"].mode()[0],
    }
    df = df.fillna(fill_vals)

    with pd.option_context("future.no_silent_downcasting", True):
        df.sex = df.sex.replace({"male": 1, "female": 0}).infer_objects(copy=False)
        df.income = df.income.replace({"under $11k": 0, "$11-$25k": 1, "$25-$50k": 2, ">$50k": 3}).infer_objects(
            copy=False
        )
    skip_cols = ["event", "sex", "time", "dzclass", "race", "diabetes", "dementia", "ca"]
    continuous_features = list(set(df.columns.to_list()).symmetric_difference(skip_cols))

    # one-hot encode categorical variables
    onehot_cols = ["dzclass", "race"]
    df = pd.get_dummies(df, columns=onehot_cols, drop_first=True)
    df = df.rename(columns={"dzclass_COPD/CHF/Cirrhosis": "dzclass_COPD"})

    df.reset_index(drop=True, inplace=True)
    df = add_prefix(df, continuous_features)
    return df


def make_nacd() -> pd.DataFrame:
    cols_to_drop = ["PERFORMANCE_STATUS", "STAGE_NUMERICAL", "AGE65"]
    df = pd.read_csv(f"{CURRENT_PATH}/NACD_Full.csv").drop(cols_to_drop, axis=1).rename(columns={"delta": "event"})

    df = df.drop(df[df["time"] <= 0].index)  # remove patients with negative or zero survival time
    df.reset_index(drop=True, inplace=True)
    continuous_features = [
        "BOX1_SCORE",
        "BOX2_SCORE",
        "BOX3_SCORE",
        "BMI",
        "WEIGHT_CHANGEPOINT",
        "AGE",
        "GRANULOCYTES",
        "LDH_SERUM",
        "LYMPHOCYTES",
        "PLATELET",
        "WBC_COUNT",
        "CALCIUM_SERUM",
        "HGB",
        "CREATININE_SERUM",
        "ALBUMIN",
    ]
    df = add_prefix(df, continuous_features)
    return df


def make_metabric() -> pd.DataFrame:
    df = pd.read_csv(f"{CURRENT_PATH}/Metabric.csv").rename(columns={"delta": "event"})
    continuous_features = ["age_at_diagnosis", "size", "lymph_nodes_positive", "stage", "lymph_nodes_removed", "NPI"]
    df = add_prefix(df, continuous_features)
    return df


def make_dlbcl() -> pd.DataFrame:
    df = pd.read_csv(f"{CURRENT_PATH}/DLBCL.csv").rename(columns={"delta": "event"})
    assert not df.isnull().values.any(), "Dataset contains NaNs"
    skip_cols = ["event", "time"]
    continuous_features = list(set(df.columns.to_list()).symmetric_difference(skip_cols))
    df = add_prefix(df, continuous_features)
    return df


def make_flchain() -> pd.DataFrame:
    # flchain dataset: relationship between serum free light chain (FLC) and mortality
    # see: https://vincentarelbundock.github.io/Rdatasets/doc/survival/flchain.html
    cols_to_drop = ["chapter"]  # only dead patients has chapter information
    df = pd.read_csv(f"{CURRENT_PATH}/flchain.csv").drop(cols_to_drop, axis=1).rename(columns={"futime": "time", "death": "event"})
    df = df.drop(df[df["time"] <= 0].index)  # remove patients with negative or zero survival time
    df.reset_index(drop=True, inplace=True)
    with pd.option_context("future.no_silent_downcasting", True):
        df.sex = df.sex.replace({"M": 1, "F": 0}).infer_objects(copy=False)
    # processing see: https://github.com/paidamoyo/adversarial_time_to_event/blob/master/data/flchain/flchain_data.py
    # data = data.fillna({"creatinine": data["creatinine"].median()})
    onehot_cols = ["sample.yr", "flc.grp"]
    df = pd.get_dummies(df, columns=onehot_cols, drop_first=True)
    skip_cols = {"event", "time", "sex", "mgus"}
    # assert not data.isnull().values.any(), "Dataset contains NaNs"
    continuous_features = list(set(df.columns.to_list()).symmetric_difference(skip_cols))
    df = add_prefix(df, continuous_features)
    return df


def make_mimic_iv_all() -> pd.DataFrame:
    df = pd.read_csv(f"{CURRENT_PATH}/MIMIC_IV_all_cause_failure.csv")
    skip_cols = ["event", "is_male", "time", "is_white", "renal", "cns", "coagulation", "cardiovascular"]
    continuous_features = list(set(df.columns.to_list()).symmetric_difference(skip_cols))
    df = add_prefix(df, continuous_features)
    return df


def make_gbsg2() -> pd.DataFrame:
    """
    German Breast Cancer Study Group (GBSG)

    This dataset is downloaded from `survival` package in R.
    The data description can be found at https://rdrr.io/cran/survival/man/gbsg.html
    """
    cols_to_drop = ["pid"]
    df = pd.read_csv(f"{CURRENT_PATH}/GBSG.csv").drop(cols_to_drop, axis=1).rename(columns={"status": "event", "rfstime": "time"})

    continuous_features = ["age", "size", "grade", "nodes", "pgr", "er"]
    df = add_prefix(df, continuous_features)
    return df


def _make_df(data):
    x = data["x"]
    t = data["t"]
    d = data["e"]

    colnames = ["x" + str(i) for i in range(x.shape[1])]
    df = pd.DataFrame(x, columns=colnames).assign(duration=t).assign(event=d)
    return df


def make_whas500() -> pd.DataFrame:
    """
    Worcester Heart Attack Study dataset. See [1] and [2] for details.
    [1] https://web.archive.org/web/20170114043458/http://www.umass.edu/statdata/statdata/data/
    [2] Hosmer, D., Lemeshow, S., May, S.: “Applied Survival Analysis: Regression Modeling of Time
    to Event Data.” John Wiley & Sons, Inc. (2008)
    """
    fn = _get_data_path(f"{CURRENT_PATH}/whas500.arff")
    df = loadarff(fn).rename(columns={"fstat": "event", "lenfol": "time"})
    df = df.astype(float)
    continuous_features = ["age", "bmi", "diasbp", "hr", "los", "sysbp"]

    df = add_prefix(df, continuous_features)
    return df


def make_valct() -> pd.DataFrame:
    """
    Veterans’ Administration Lung Cancer Trial

    [1] Kalbfleisch, J.D., Prentice, R.L.: “The Statistical Analysis of Failure Time Data.”
    John Wiley & Sons, Inc. (2002)
    """
    fn = _get_data_path(f"{CURRENT_PATH}/veteran.arff")
    df = loadarff(fn).rename(columns={"Status": "event", "Survival_in_days": "time"})
    with pd.option_context("future.no_silent_downcasting", True):
        df.event = df.event.replace({'dead': 1, 'censored': 0}).infer_objects(copy=False)
        df.Prior_therapy = df.Prior_therapy.replace({'no': 0, 'yes': 1}).infer_objects(copy=False)
        df.Treatment = df.Treatment.replace({'standard': 0, 'test': 1}).infer_objects(copy=False)
    df = pd.get_dummies(df, columns=['Celltype'], drop_first=True)
    df = df.astype(float)

    continuous_features = ['Age_in_years', 'Karnofsky_score', 'Months_from_Diagnosis']
    df = add_prefix(df, continuous_features)

    return df


def make_gbm() -> pd.DataFrame:
    df = pd.read_csv(f"{CURRENT_PATH}/GBM.clin.merged.picked.csv").rename(columns={"delta": "event"})
    df.drop(columns=["Composite Element REF", "tumor_tissue_site"], inplace=True)  # Columns with only one value
    df = df[df.time.notna()]  # Unknown censor/event time
    df = df.drop(df[df["time"] <= 0].index)  # remove patients with negative or zero survival time
    df.reset_index(drop=True, inplace=True)

    # Preprocess and fill missing values
    with pd.option_context("future.no_silent_downcasting", True):
        df.gender = df.gender.replace({"male": 1, "female": 0}).infer_objects(copy=False)
        df.radiation_therapy = df.radiation_therapy.replace({"yes": 1, "no": 0}).infer_objects(copy=False)
        df.ethnicity = df.ethnicity.replace({"not hispanic or latino": 0, "hispanic or latino": 1}).infer_objects(
            copy=False
        )
    # one-hot encode categorical variables
    onehot_cols = ["histological_type", "race"]
    df = pd.get_dummies(df, columns=onehot_cols, drop_first=True)
    # fill_vals = {
    #     "radiation_therapy": data["radiation_therapy"].median(),
    #     "karnofsky_performance_score": data["karnofsky_performance_score"].median(),
    #     "ethnicity": data["ethnicity"].median()
    # }
    # data = data.fillna(fill_vals)
    df.columns = df.columns.str.replace(" ", "_")

    continuous_features = ["years_to_birth", "date_of_initial_pathologic_diagnosis", "karnofsky_performance_score"]
    df = add_prefix(df, continuous_features)
    return df


def make_seer_liver() -> pd.DataFrame:
    """
    Preprocess the SEER liver cancer dataset.
    """
    df = pd.read_csv(f"{CURRENT_PATH}/SEER/Liver.csv").rename(columns={"Survival months": "time"})
    df = df.drop(df[df["time"] <= 0].index)  # remove patients with negative or zero survival time
    df.reset_index(drop=True, inplace=True)

    skip_cols = ["event", "time", "Sex"]
    continuous_features = list(set(df.columns.to_list()).symmetric_difference(skip_cols))
    df = add_prefix(df, continuous_features)

    return df


def make_seer_brain() -> pd.DataFrame:
    """
    Preprocess the SEER brain cancer dataset.
    """
    df = pd.read_csv(f"{CURRENT_PATH}/SEER/Brain.csv").rename(columns={"Survival months": "time"})
    df = df.drop(df[df["time"] <= 0].index)  # remove patients with negative or zero survival time
    df.reset_index(drop=True, inplace=True)

    skip_cols = [
        "event",
        "time",
        "Sex",
        "Behavior recode for analysis",
        "SEER historic stage A (1973-2015)",
        "RX Summ--Scope Reg LN Sur (2003+)",
    ]
    continuous_features = list(set(df.columns.to_list()).symmetric_difference(skip_cols))
    df = add_prefix(df, continuous_features)

    return df


def make_seer_stomach() -> pd.DataFrame:
    """
    Preprocess the SEER stomach cancer dataset.
    """
    df = pd.read_csv(f"{CURRENT_PATH}/SEER/Stomach.csv").rename(columns={"Survival months": "time"})
    df = df.drop(df[df["time"] <= 0].index)  # remove patients with negative or zero survival time
    df.reset_index(drop=True, inplace=True)

    skip_cols = ["event", "time", "Sex"]
    continuous_features = list(set(df.columns.to_list()).symmetric_difference(skip_cols))
    df = add_prefix(df, continuous_features)

    return df


def make_pbc() -> pd.DataFrame:
    """
    Preprocess the Cirrhosis Patient Survival Prediction dataset.

    Link: https://archive.ics.uci.edu/dataset/878/cirrhosis+patient+survival+prediction+dataset-1
    Paper: https://pubmed.ncbi.nlm.nih.gov/2737595/
    """
    with open(f"{CURRENT_PATH}/cirrhosis.pkl", "rb") as f:
        cirrhosis = pickle.load(f)

    cols_to_drop = ["ID"]
    df = cirrhosis.data.original.drop(cols_to_drop, axis=1).rename(columns={"Status": "event", "N_Days": "time"})
    with pd.option_context("future.no_silent_downcasting", True):
        df = df.replace({"NaNN": np.nan}).infer_objects(copy=False)
        df.event = df.event.replace({"C": 0, "CL": 0, "D": 1}).infer_objects(copy=False)
        df.Drug = df.Drug.replace({"D-penicillamine": 0, "Placebo": 1}).infer_objects(copy=False)
        df.Sex = df.Sex.replace({"M": 1, "F": 0}).infer_objects(copy=False)
        df.Ascites = df.Ascites.replace({"N": 0, "Y": 1}).infer_objects(copy=False)
        df.Hepatomegaly = df.Hepatomegaly.replace({"N": 0, "Y": 1}).infer_objects(copy=False)
        df.Spiders = df.Spiders.replace({"N": 0, "Y": 1}).infer_objects(copy=False)
        df.Edema = df.Edema.replace({"N": 0, "Y": 1, "S": 0.5}).infer_objects(copy=False)
    df.Cholesterol = pd.to_numeric(df.Cholesterol, errors="coerce")
    df.Copper = pd.to_numeric(df.Copper, errors="coerce")
    df.Tryglicerides = pd.to_numeric(df.Tryglicerides, errors="coerce")
    df.Platelets = pd.to_numeric(df.Platelets, errors="coerce")

    # fill_vals = {
    #     "Drug": data.Drug.mode()[0],
    #     "Ascites": data.Ascites.mode()[0],
    #     "Hepatomegaly": data.Hepatomegaly.mode()[0],
    #     "Spiders": data.Spiders.mode()[0],
    #     "Cholesterol": data.Cholesterol.mean(),
    #     "Copper": data.Copper.mean(),
    #     "Alk_Phos": data.Alk_Phos.mean(),
    #     "SGOT": data.SGOT.mean(),
    #     "Tryglicerides": data.Tryglicerides.mean(),
    #     "Platelets": data.Platelets.mean(),
    #     "Prothrombin": data.Prothrombin.mean(),
    #     "Stage": data.Stage.mode()[0],
    # }
    #
    # data = data.fillna(fill_vals)
    df.reset_index(drop=True, inplace=True)

    skip_cols = ["Drug", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema", "Stage", "event", "time"]
    continuous_features = list(set(df.columns.to_list()).symmetric_difference(skip_cols))
    df = add_prefix(df, continuous_features)
    return df


def make_heart_failure() -> pd.DataFrame:
    """
    Preprocess the Heart Failure Prediction dataset.

    Link: https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records
    Paper: https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5
    """
    with open(f"{CURRENT_PATH}/heart_failure.pkl", "rb") as f:
        heart_failure = pickle.load(f)

    X = heart_failure.data.features
    y = heart_failure.data.targets
    df = pd.concat([X, y], axis=1).rename(columns={"death_event": "event"})

    continuous_features = [
        "age",
        "creatinine_phosphokinase",
        "ejection_fraction",
        "platelets",
        "serum_creatinine",
        "serum_sodium",
    ]
    df = add_prefix(df, continuous_features)

    return df


def make_churn() -> pd.DataFrame:
    """
    Predicting when your customers will churn.

    Data description: https://square.github.io/pysurvival/tutorials/churn.html
    Data downloaded from PySurvival: https://github.com/square/pysurvival/tree/master/pysurvival/datasets

    Link: https://www.kaggle.com/blastchar/telco-customer-churn
    """
    churn = pd.read_csv(f"{CURRENT_PATH}/churn.csv").rename(columns={"months_active": "time", "churned": "event"})
    churn.event = churn.event.astype(int)

    with pd.option_context("future.no_silent_downcasting", True):
        churn.product_travel_expense = churn.product_travel_expense.replace(
            {"No": 0, "Free-Trial": 1, "Active": 2}
        ).infer_objects(copy=False)
        churn.product_payroll = churn.product_payroll.replace({"No": 0, "Free-Trial": 1, "Active": 2}).infer_objects(
            copy=False
        )
        churn.product_accounting = churn.product_accounting.replace(
            {"No": 0, "Free-Trial": 1, "Active": 2}
        ).infer_objects(copy=False)
        churn.company_size = churn.company_size.replace(
            {
                "self-employed": 0,
                "1-10": 1,
                "10-50": 2,
                "50-100": 3,
                "100-250": 4,
            }
        ).infer_objects(copy=False)
    # creating one-hot vectors for categorical variables
    cat_cols = ["us_region"]
    df = pd.get_dummies(churn, columns=cat_cols, drop_first=True)

    df = df.drop(df[df["time"] <= 0].index)  # remove patients with negative or zero survival time
    df.reset_index(drop=True, inplace=True)

    continuous_features = [
        "product_data_storage",
        "product_travel_expense",
        "product_payroll",
        "product_accounting",
        "csat_score",
        "articles_viewed",
        "marketing_emails_clicked",
        "minutes_customer_support",
        "company_size",
    ]
    df = add_prefix(df, continuous_features)

    return df


def make_employee_retention() -> pd.DataFrame:
    """
    Predicting employee retention.

    Data description:
        https://square.github.io/pysurvival/tutorials/employee_retention.html
    Data downloaded from PySurvival:
        https://github.com/square/pysurvival/blob/master/pysurvival/datasets/employee_attrition.csv
    """
    retention = pd.read_csv(f"{CURRENT_PATH}/employee_attrition.csv").rename(
        columns={"time_spend_company": "time", "left": "event"}
    )

    with pd.option_context("future.no_silent_downcasting", True):
        retention.salary = retention.salary.replace({"low": 0, "medium": 1, "high": 2}).infer_objects(copy=False)
    # creating one-hot vectors for categorical variables
    cat_cols = ["department"]
    df = pd.get_dummies(retention, columns=cat_cols, drop_first=True)

    df = df.drop_duplicates(keep="first").reset_index(drop=True)

    continuous_features = [
        "satisfaction_level",
        "last_evaluation",
        "number_projects",
        "average_montly_hours",
        "work_accident",
        "promotion_last_5years",
        "salary",
    ]

    df = add_prefix(df, continuous_features)

    return df


def make_pdm() -> pd.DataFrame:
    """
    Predictive maintenance dataset.

    Data description:
        https://square.github.io/pysurvival/tutorials/maintenance.html
    Data downloaded from PySurvival:
        https://github.com/square/pysurvival/blob/master/pysurvival/datasets/maintenance.csv
    """
    pdm = pd.read_csv(f"{CURRENT_PATH}/maintenance.csv", sep=";").rename(columns={"lifetime": "time", "broken": "event"})

    cat_cols = ["team", "provider"]
    df = pd.get_dummies(pdm, columns=cat_cols, drop_first=True)

    continuous_features = [
        "pressureInd",
        "moistureInd",
        "temperatureInd",
    ]

    df = add_prefix(df, continuous_features)

    return df
