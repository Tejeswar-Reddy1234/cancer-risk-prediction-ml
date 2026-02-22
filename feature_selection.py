import numpy as np
import config
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


def chi_square_selection(X_train, y_train, X_test):

    print("Running Chi-Square Feature Selection...")

    selector = SelectKBest(chi2, k=config.N_FEATURES)

    X_train_new = selector.fit_transform(np.abs(X_train), y_train)
    X_test_new = selector.transform(np.abs(X_test))

    return X_train_new, X_test_new


def mutual_info_selection(X_train, y_train, X_test):

    print("Running Mutual Information Feature Selection...")

    selector = SelectKBest(mutual_info_classif, k=config.N_FEATURES)

    X_train_new = selector.fit_transform(X_train, y_train)
    X_test_new = selector.transform(X_test)

    return X_train_new, X_test_new


def rfe_selection(X_train, y_train, X_test):

    print("Running RFE Feature Selection...")

    model = LogisticRegression(max_iter=500)

    selector = RFE(model, n_features_to_select=config.N_FEATURES)

    X_train_new = selector.fit_transform(X_train, y_train)
    X_test_new = selector.transform(X_test)

    return X_train_new, X_test_new