import pandas as pd
from sklearn.datasets import load_breast_cancer

def load_data():

    print("Loading Breast Cancer Dataset...")

    data = load_breast_cancer()

    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    return X, y, data.feature_names