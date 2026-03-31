import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.utils.validation import check_is_fitted
import copy
from config import MODEL_CONFIGS

def _build_base_estimator(model_type, params):
    """
    Lightweight internal estimator factory for MoE experts.

    We keep this local to avoid circular imports with modeling.py.
    """
    if model_type == "logistic_regression":
        return LogisticRegression(**params)

    if model_type == "elastic_net_logistic":
        return LogisticRegression(**params)

    if model_type == "random_forest":
        return RandomForestClassifier(**params)

    if model_type == "xgboost":
        return XGBClassifier(**params)
    
    if model_name == "mixture_of_experts":
        raise ValueError("mixture_of_experts cannot be used as its own expert.")

    raise ValueError(f"Unsupported expert model_type: {model_type}")

class MixtureOfExpertsClassifier(BaseEstimator, ClassifierMixin):
    """
    Soft-gated mixture-of-experts classifier for binary classification.

    Workflow
    --------
    1. Fit each expert on the same preprocessed feature matrix X.
    2. Get each expert's predicted probability p_k(x).
    3. Fit a gating model g(x) that outputs expert weights w_k(x),
       where weights sum to 1 across experts for each row.
    4. Final prediction:
           p(y=1 | x) = sum_k w_k(x) * p_k(x)

    Notes
    -----
    - Designed to plug into an sklearn Pipeline as the final estimator.
    - Assumes the incoming X is already numeric/preprocessed.
    - Uses soft gating, not hard routing.
    """

    def __init__(
        self,
        expert_model_names=None,
        gate_C=1.0,
        random_state=42,
    ):
        self.expert_model_names = expert_model_names
        self.gate_C = gate_C
        self.random_state = random_state
        
    def _build_default_experts(self):
        """
        Build experts from MODEL_CONFIGS so hyperparameters live in one place.
        """
        expert_names = self.expert_model_names or [
            "logistic_regression",
            "random_forest",
            "elastic_net_logistic",
            "xgboost",
        ]

        experts = []
        for model_name in expert_names:
            if model_name not in MODEL_CONFIGS:
                raise ValueError(
                    f"Expert '{model_name}' not found in MODEL_CONFIGS. "
                    f"Available: {list(MODEL_CONFIGS.keys())}"
                )

            model_spec = MODEL_CONFIGS[model_name]
            model_type = model_spec["type"]
            params = copy.deepcopy(model_spec.get("params", {}))

            # Force/override shared random state where appropriate
            if "random_state" in params:
                params["random_state"] = self.random_state

            # Keep xgboost stable if not already specified
            if model_type == "xgboost":
                params.setdefault("eval_metric", "logloss")
                params.setdefault("n_jobs", -1)

            estimator = _build_base_estimator(model_type, params)
            experts.append((model_name, estimator))

        return experts

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).astype(int)

        self.classes_ = np.array([0, 1])

        # Always build actual estimators here
        expert_specs = self._build_default_experts()

        self.experts_ = []
        expert_probas = []

        for name, expert in expert_specs:
            fitted = expert.fit(X, y)
            self.experts_.append((name, fitted))
            p = fitted.predict_proba(X)[:, 1]
            expert_probas.append(p)

        expert_probas = np.column_stack(expert_probas)

        expert_errors = np.abs(expert_probas - y.reshape(-1, 1))
        gate_target = np.argmin(expert_errors, axis=1)

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
        X = np.asarray(X)
        probs = []
        for _, expert in self.experts_:
            probs.append(expert.predict_proba(X)[:, 1])
        return np.column_stack(probs)

    def predict_proba(self, X):
        check_is_fitted(self, ["experts_", "gate_", "n_experts_"])

        X = np.asarray(X)
        expert_probas = self._expert_matrix(X)              # (n, k)
        gate_weights = self.gate_.predict_proba(X)          # (n, k)

        # If a class is absent in gate training for edge cases, align columns.
        if gate_weights.shape[1] != self.n_experts_:
            full = np.zeros((X.shape[0], self.n_experts_))
            for j, cls in enumerate(self.gate_.classes_):
                full[:, int(cls)] = gate_weights[:, j]
            gate_weights = full

        final_p1 = np.sum(gate_weights * expert_probas, axis=1)
        final_p1 = np.clip(final_p1, 1e-8, 1 - 1e-8)

        return np.column_stack([1 - final_p1, final_p1])

    def predict(self, X):
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= 0.5).astype(int)

    @property
    def feature_importances_(self):
        """
        Fallback importances so your existing importance extractor
        won't crash. This is not a true MoE-global importance measure.
        We return the RF expert's importance if available, otherwise
        raise AttributeError so downstream code can skip it.
        """
        for name, expert in self.experts_:
            if hasattr(expert, "feature_importances_"):
                return expert.feature_importances_
        raise AttributeError("MoE does not expose a single native feature_importances_ vector.")

    @property
    def coef_(self):
        """
        Fallback coefficient exposure using the gating model coefficients.
        This is not equivalent to global model coefficients, but it lets
        linear-style tooling inspect gate behavior if needed.
        """
        if hasattr(self.gate_, "coef_"):
            # Flatten so downstream code expecting 1d coef_ does not explode.
            return self.gate_.coef_.mean(axis=0, keepdims=True)
        raise AttributeError("MoE gate does not expose coef_.")