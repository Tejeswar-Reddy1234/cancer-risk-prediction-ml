import config
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def get_models():

    return {
        "LogisticRegression": LogisticRegression(max_iter=500),
        "SVM": SVC(probability=True),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            random_state=config.RANDOM_STATE
        )
    }