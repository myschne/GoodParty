"""
Mixture of Experts model implementation for the Candidate Success pipeline.

This module defines a custom sklearn-compatible classifier that combines
multiple base models into a single soft-gated ensemble. Each expert model
is fit on the same preprocessed feature matrix, and a gating model learns
how to weight the experts' predicted probabilities for each observation.

Main responsibilities of this module:
- build expert estimators from MODEL_CONFIGS
- fit each expert on the training data
- learn a gating model to combine expert predictions
- return sklearn-compatible predict and predict_proba outputs
- expose limited fallback coefficient / importance attributes for downstream compatibility

This file is designed to integrate with the existing project pipeline so the
Mixture of Experts model can be trained, evaluated, and scored using the same
workflow as the other supported models.
"""

import copy
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier
from scipy import sparse

from config import MODEL_CONFIGS


# =========================================================
# Expert estimator factory
# =========================================================

def _build_base_estimator(model_type, params):
    """
    Build one expert estimator from a model type and parameter dictionary.

    This local helper mirrors the supported model families used elsewhere
    in the project, but is defined here to avoid circular imports with
    modeling.py.

    Parameters
    ----------
    model_type : str
        Internal model family name.
    params : dict
        Hyperparameters used to initialize the estimator.

    Returns
    -------
    estimator
        Unfit sklearn-compatible estimator for use as an MoE expert.

    Raises
    ------
    ValueError
        If the requested model type is not supported as an MoE expert.
    """
    if model_type == "logistic_regression":
        return LogisticRegression(**params)

    if model_type == "elastic_net_logistic":
        return LogisticRegression(**params)

    if model_type == "random_forest":
        return RandomForestClassifier(**params)

    if model_type == "xgboost":
        return XGBClassifier(**params)

    raise ValueError(f"Unsupported expert model_type: {model_type}")


# =========================================================
# Mixture of Experts classifier
# =========================================================

class MixtureOfExpertsClassifier(ClassifierMixin, BaseEstimator):
    """
    Soft-gated Mixture of Experts classifier for binary classification.

    Workflow
    --------
    1. Fit each expert model on the same preprocessed feature matrix X.
    2. Collect each expert's predicted probability p_k(x).
    3. Identify which expert was closest to the observed label for each row.
    4. Fit a multinomial logistic gating model to predict expert weights.
    5. Compute the final probability as the weighted average of expert probabilities.

    Notes
    -----
    - Designed to plug into an sklearn Pipeline as the final estimator.
    - Assumes the incoming X is already numeric / preprocessed.
    - Uses soft gating rather than hard routing.
    """
    _estimator_type = "classifier"
    def __init__(
        self,
        expert_model_names=None,
        gate_C=1.0,
        random_state=42,
    ):
        """
        Initialize the Mixture of Experts classifier.

        Parameters
        ----------
        expert_model_names : list[str] or None, default=None
            List of MODEL_CONFIGS keys to use as experts. If None, a default
            expert set is used.
        gate_C : float, default=1.0
            Inverse regularization strength for the multinomial logistic
            gating model. Smaller values imply stronger regularization.
        random_state : int, default=42
            Random seed used where supported for reproducibility.
        """
        self.expert_model_names = expert_model_names
        self.gate_C = gate_C
        self.random_state = random_state
    
    def _prepare_X(self, X):
        """
        Preserve sparse matrices as CSR and coerce dense inputs to ndarray.
        """
        if sparse.issparse(X):
            return X.tocsr()
        return np.asarray(X)

    def _build_default_experts(self):
        """
        Build the configured expert estimators from MODEL_CONFIGS.

        This keeps expert selection and hyperparameters centralized in
        config.py instead of hard-coding them inside the class.

        Returns
        -------
        list[tuple[str, estimator]]
            List of (expert_name, unfit_estimator) pairs.

        Raises
        ------
        ValueError
            If an expert name is missing from MODEL_CONFIGS or if the MoE
            model is mistakenly included as its own expert.
        """
        expert_names = self.expert_model_names or [
            "logistic_regression",
            "random_forest",
            "elastic_net_logistic",
            "xgboost",
        ]

        experts = []
        for model_name in expert_names:
            if model_name == "mixture_of_experts":
                raise ValueError("mixture_of_experts cannot be used as its own expert.")

            if model_name not in MODEL_CONFIGS:
                raise ValueError(
                    f"Expert '{model_name}' not found in MODEL_CONFIGS. "
                    f"Available: {list(MODEL_CONFIGS.keys())}"
                )

            model_spec = MODEL_CONFIGS[model_name]
            model_type = model_spec["type"]
            params = copy.deepcopy(model_spec.get("params", {}))

            # Override shared random-state parameters where applicable.
            if "random_state" in params:
                params["random_state"] = self.random_state

            # Ensure stable binary-classification defaults for XGBoost experts.
            if model_type == "xgboost":
                params.setdefault("eval_metric", "logloss")
                params.setdefault("n_jobs", -1)

            estimator = _build_base_estimator(model_type, params)
            experts.append((model_name, estimator))

        return experts

    def fit(self, X, y):
        """
        Fit all experts and then fit the gating model.

        Parameters
        ----------
        X : array-like
            Preprocessed feature matrix.
        y : array-like
            Binary target labels.

        Returns
        -------
        self
            Fitted MixtureOfExpertsClassifier instance.

        Notes
        -----
        The gating target is defined as the expert whose predicted
        probability is closest to the observed label for each training row.
        """
        X = self._prepare_X(X)
        y = np.asarray(y).astype(int)

        # Store sklearn-compatible class labels.
        self.classes_ = np.array([0, 1])

        # Build and fit all configured expert estimators.
        expert_specs = self._build_default_experts()

        self.experts_ = []
        expert_probas = []

        for name, expert in expert_specs:
            fitted = expert.fit(X, y)
            self.experts_.append((name, fitted))
            p = fitted.predict_proba(X)[:, 1]
            expert_probas.append(p)

        # Stack expert probabilities into shape (n_samples, n_experts).
        expert_probas = np.column_stack(expert_probas)

        # Define the gating target as the expert with the smallest absolute
        # prediction error for each training observation.
        expert_errors = np.abs(expert_probas - y.reshape(-1, 1))
        gate_target = np.argmin(expert_errors, axis=1)

        # Fit the multinomial gating model that learns expert weights.
        self.gate_ = LogisticRegression(
            multi_class="multinomial",
            max_iter=2000,
            C=self.gate_C,
            random_state=self.random_state,
        )
        self.gate_.fit(X, gate_target)

        self.n_experts_ = expert_probas.shape[1]
        return self

    def _expert_matrix(self, X):
        """
        Generate the matrix of expert predicted probabilities.

        Parameters
        ----------
        X : array-like
            Preprocessed feature matrix.

        Returns
        -------
        np.ndarray
            Matrix of shape (n_samples, n_experts) containing class-1
            predicted probabilities from each fitted expert.
        """
        X = self._prepare_X(X)
        probs = []
        for _, expert in self.experts_:
            probs.append(expert.predict_proba(X)[:, 1])
        return np.column_stack(probs)

    def predict_proba(self, X):
        """
        Predict class probabilities using the soft-gated expert ensemble.

        Parameters
        ----------
        X : array-like
            Preprocessed feature matrix.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, 2) with class-0 and class-1 probabilities.

        Notes
        -----
        If one or more expert classes are absent in gate training, the gating
        output columns are realigned to the expected expert index space.
        """
        check_is_fitted(self, ["experts_", "gate_", "n_experts_"])

        X = self._prepare_X(X)
        expert_probas = self._expert_matrix(X)
        gate_weights = self.gate_.predict_proba(X)

        # If some expert class labels are missing in gate training,
        # realign the gate output to the full expert index set.
        if gate_weights.shape[1] != self.n_experts_:
            full = np.zeros((X.shape[0], self.n_experts_))
            for j, cls in enumerate(self.gate_.classes_):
                full[:, int(cls)] = gate_weights[:, j]
            gate_weights = full

        # Weighted average of expert probabilities.
        final_p1 = np.sum(gate_weights * expert_probas, axis=1)

        # Clip for numerical stability.
        final_p1 = np.clip(final_p1, 1e-8, 1 - 1e-8)

        return np.column_stack([1 - final_p1, final_p1])

    def predict(self, X):
        """
        Predict binary class labels.

        Parameters
        ----------
        X : array-like
            Preprocessed feature matrix.

        Returns
        -------
        np.ndarray
            Predicted binary labels using a 0.5 probability threshold.
        """
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= 0.5).astype(int)

    @property
    def feature_importances_(self):
        """
        Provide a fallback feature-importance vector for compatibility.

        Returns
        -------
        np.ndarray
            Feature importances from the first fitted expert that exposes
            `feature_importances_`, typically a tree-based model.

        Raises
        ------
        AttributeError
            If no fitted expert exposes feature importances.

        Notes
        -----
        This is not a true global MoE importance measure. It exists only so
        downstream code that expects a feature_importances_ attribute does
        not fail immediately.
        """
        for name, expert in self.experts_:
            if hasattr(expert, "feature_importances_"):
                return expert.feature_importances_
        raise AttributeError("MoE does not expose a single native feature_importances_ vector.")

    @property
    def coef_(self):
        """
        Provide a fallback coefficient vector for compatibility.

        Returns
        -------
        np.ndarray
            Mean gating-model coefficients reshaped to a 2D array.

        Raises
        ------
        AttributeError
            If the fitted gating model does not expose coefficients.

        Notes
        -----
        This is not equivalent to a single global coefficient vector for the
        entire Mixture of Experts model. It is only a fallback representation
        of gating behavior for downstream linear-model tooling.
        """
        if hasattr(self.gate_, "coef_"):
            return self.gate_.coef_.mean(axis=0, keepdims=True)
        raise AttributeError("MoE gate does not expose coef_.")