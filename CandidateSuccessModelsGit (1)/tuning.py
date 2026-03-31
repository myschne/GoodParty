import pandas as pd
from sklearn.model_selection import GridSearchCV, GroupKFold

from config import MODEL_CONFIGS, N_FOLDS
from feature_engineering import split_X_y
from modeling import make_model_pipeline


PARAM_GRIDS = {
    "logistic_regression": {
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__class_weight": [None, "balanced"],
    },

    "elastic_net_logistic": {
        "model__C": [0.01, 0.1, 1.0],
        "model__l1_ratio": [0.2, 0.5, 0.8],
        "model__class_weight": [None, "balanced"],
    },

    "random_forest": {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
    },

    "xgboost": {
        "model__n_estimators": [200, 300],
        "model__max_depth": [3, 4, 6],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
        "model__reg_alpha": [0.0, 0.1],
        "model__reg_lambda": [1.0, 5.0],
    },
}


def make_group_id(df: pd.DataFrame) -> pd.Series:
    """
    Match the same group logic used in cv.py:
    hubspot_id | election_date
    """
    return (
        df["hubspot_id"].astype(str).str.strip() + "|" +
        df["election_date"].astype(str)
    )


def run_grid_search(full_df, model_name, scoring="roc_auc", n_jobs=-1, verbose=2):
    """
    Run grouped GridSearchCV for one model in MODEL_CONFIGS.
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model_name: {model_name}")

    if model_name not in PARAM_GRIDS:
        raise ValueError(f"No parameter grid defined for model_name: {model_name}")

    model_spec = MODEL_CONFIGS[model_name]
    model_type = model_spec["type"]
    model_params = model_spec["params"]

    # Build features/target using your existing feature engineering
    X, y, _ = split_X_y(full_df, target_col="Win")

    # Build grouped CV ids exactly like your project logic
    groups = make_group_id(full_df)
    cv = GroupKFold(n_splits=N_FOLDS)

    # Build your existing preprocessing + estimator pipeline
    clf, _, _ = make_model_pipeline(
        X,
        model_type=model_type,
        params=model_params,
    )

    grid = GridSearchCV(
        estimator=clf,
        param_grid=PARAM_GRIDS[model_name],
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True,
        return_train_score=True,
    )

    grid.fit(X, y, groups=groups)

    results_df = pd.DataFrame(grid.cv_results_).sort_values(
        by="rank_test_score"
    ).reset_index(drop=True)

    return {
        "model_name": model_name,
        "model_type": model_type,
        "best_score": grid.best_score_,
        "best_params": grid.best_params_,
        "best_estimator": grid.best_estimator_,
        "cv_results_df": results_df,
        "grid_search": grid,
    }


def run_grid_search_all_models(full_df, model_names=None, scoring="roc_auc", n_jobs=-1, verbose=2):
    """
    Run grouped GridSearchCV for multiple models and compare their best results.
    """
    if model_names is None:
        model_names = list(PARAM_GRIDS.keys())

    summaries = []
    full_results = {}

    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"Running grid search for {model_name}")
        print(f"{'='*60}")

        out = run_grid_search(
            full_df=full_df,
            model_name=model_name,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        summaries.append({
            "model_name": out["model_name"],
            "model_type": out["model_type"],
            "best_score": out["best_score"],
            "best_params": out["best_params"],
        })

        full_results[model_name] = out

        print(f"Best score for {model_name}: {out['best_score']:.4f}")
        print(f"Best params: {out['best_params']}")

    summary_df = pd.DataFrame(summaries).sort_values(
        by="best_score",
        ascending=False
    ).reset_index(drop=True)

    return {
        "summary_df": summary_df,
        "full_results": full_results,
    }