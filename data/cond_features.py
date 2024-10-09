import numpy as np
import torch
from functools import partial

from data import make_survival_data

cond_features = {
    "VALCT": ["Age_in_years"],
    "DLBCL": ['f0001'],
    'HFCR': ['age', 'sex'],
    "PBC": ['Age', 'Sex', ],
    'WHAS500': ['age', 'gender', 'bmi', ],
    "GBM": ['years_to_birth', 'gender'],
    'GBSG': ['age', 'hormon'],  # hormon: hormonal therapy
    "PDM": ['temperatureInd'],
    "METABRIC": ['age_at_diagnosis'],
    "churn": ['company_size'],
    "NACD": ['AGE', 'GENDER', 'BMI', ],
    "FLCHAIN": ['age', 'sex'],
    "SUPPORT": ['age', 'sex'],
    'employee': ['salary'],
    'MIMIC-IV_all': ['age', 'is_male', 'is_white'],
    'SEER_brain': ['Age recode with single ages and 85+', 'Sex'],
    'SEER_liver': ['Age recode with single ages and 85+', 'Sex'],
    'SEER_stomach': ['Age recode with single ages and 85+', 'Sex'],
}

categorical_features = [
    'Recipientgender',
    'sex',
    'drug',
    'gender',
    'treat',
    'is_male',
    'is_white',
    'male',
    'company_size',
    'hormon',
]
categorical_features = [f.lower() for f in categorical_features]
categorical_features = list(set(categorical_features))


def get_cond_functions(dataset: str) -> list:
    """Get the conditional functions for the dataset."""
    cond_functions = []
    for cond_feat in cond_features[dataset]:
        if cond_feat.lower() in categorical_features:
            # get unique values of the categorical feature
            data = make_survival_data(dataset)[0]
            unique_values = np.unique(data[cond_feat])
            for v in unique_values:
                cond_functions.append(partial(lambda x, feat=cond_feat, val=v:
                                              (x[:, get_cond_idx(dataset, feat)] == val).astype(int)))
            # cond_functions.append(partial(lambda x, feat=cond_feat: x[:, get_cond_idx(dataset, feat)]))
        else:
            # for continuous features, we use 0 as the threshold, it is because the features are normalized.
            cond_functions.append(partial(lambda x, feat=cond_feat:
                                          (x[:, get_cond_idx(dataset, feat)] > 0).astype(int)))
            cond_functions.append(partial(lambda x, feat=cond_feat:
                                          (x[:, get_cond_idx(dataset, feat)] <= 0).astype(int)))
    return cond_functions


def get_cond_idx(dataset: str, cond_feature: str):
    """get the index of the conditional features in the dataset, without the time and event columns."""
    data, _ = make_survival_data(dataset)
    features = data.drop(columns=['time', 'event']).columns
    return features.get_loc(cond_feature)


if __name__ == '__main__':
    for data_name in cond_features:
        data, _ = make_survival_data(data_name)
        features = data.columns.to_list()
        cond_feat = cond_features[data_name]
        if cond_feat is not None:
            for f in cond_feat:
                if f not in features:
                    raise ValueError(f"Feature {f} not found in the dataset {data_name}.")

    print("All conditional features are found in the datasets.")
