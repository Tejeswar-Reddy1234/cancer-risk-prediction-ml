print("TRAIN PIPELINE STARTED")

import joblib
import config

from src.dataloader import load_data
from src.preprocessing import split_and_scale
from src.feature_selection import (
    chi_square_selection,
    mutual_info_selection,
    rfe_selection
)
from src.models import get_models
from src.evaluation import evaluate


def run_pipeline():

    print("Loading data...")
    X, y, feature_names = load_data()

    print("Preprocessing...")
    X_train, X_test, y_train, y_test = split_and_scale(X, y)

    print("Applying Feature Selection...")

    feature_sets = {
        "CHI": chi_square_selection(X_train, y_train, X_test),
        "MI": mutual_info_selection(X_train, y_train, X_test),
        "RFE": rfe_selection(X_train, y_train, X_test),
    }

    models = get_models()

    best_model = None
    best_score = 0

    for fs_name, (Xtr, Xte) in feature_sets.items():

        print(f"\n=== Feature Selection: {fs_name} ===")

        for model_name, model in models.items():

            model.fit(Xtr, y_train)

            metrics = evaluate(model, Xte, y_test)

            print(model_name, metrics)

            if metrics["accuracy"] > best_score:
                best_score = metrics["accuracy"]
                best_model = model

    joblib.dump(best_model, config.MODEL_PATH)

    print("\nBest model saved!")