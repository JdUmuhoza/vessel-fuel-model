"""Baseline, pure-ML, physics-only, and hybrid residual models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from vessel_fuel.model import fuel_model


class SpeedPowerBaseline:
    r"""Simple speed-power baseline: $F = a\,d\,v^b$."""

    def __init__(self) -> None:
        self.a = 1.0
        self.b = 3.0

    def fit(self, observations: Sequence[dict[str, Any]]) -> "SpeedPowerBaseline":
        d = np.asarray([float(o["distance_km"]) for o in observations], dtype=float)
        v = np.asarray([max(float(o["speed_tw_kn"]), 1e-3) for o in observations], dtype=float)
        y = np.asarray([max(float(o["fuel_mt"]), 1e-6) for o in observations], dtype=float)

        # log(y / d) = log(a) + b log(v)
        x = np.log(v)
        t = np.log(y / np.maximum(d, 1e-6))
        A = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(A, t, rcond=None)
        self.a = float(np.exp(coef[0]))
        self.b = float(coef[1])
        return self

    def predict(self, observations: Sequence[dict[str, Any]]) -> np.ndarray:
        d = np.asarray([float(o["distance_km"]) for o in observations], dtype=float)
        v = np.asarray([max(float(o["speed_tw_kn"]), 1e-3) for o in observations], dtype=float)
        return self.a * d * (v**self.b)


class CubicLawModel:
    r"""Fixed cubic law baseline: $F = a\,d\,v^3$."""

    def __init__(self) -> None:
        self.a = 1.0

    def fit(self, observations: Sequence[dict[str, Any]]) -> "CubicLawModel":
        d = np.asarray([max(float(o["distance_km"]), 1e-6) for o in observations], dtype=float)
        v = np.asarray([max(float(o["speed_tw_kn"]), 1e-3) for o in observations], dtype=float)
        y = np.asarray([max(float(o["fuel_mt"]), 1e-9) for o in observations], dtype=float)

        x = d * (v**3)
        denom = float(np.dot(x, x))
        self.a = float(np.dot(x, y) / denom) if denom > 0.0 else 1.0
        return self

    def predict(self, observations: Sequence[dict[str, Any]]) -> np.ndarray:
        d = np.asarray([max(float(o["distance_km"]), 0.0) for o in observations], dtype=float)
        v = np.asarray([max(float(o["speed_tw_kn"]), 0.0) for o in observations], dtype=float)
        return np.maximum(self.a * d * (v**3), 0.0)


class PhysicsOnlyModel:
    """Wrapper around physics-based fuel model."""

    def __init__(self, calib: dict[str, float] | None = None) -> None:
        self.calib = calib

    def fit(self, observations: Sequence[dict[str, Any]]) -> "PhysicsOnlyModel":
        _ = observations
        return self

    def predict(self, observations: Sequence[dict[str, Any]]) -> np.ndarray:
        return np.asarray(
            [
                float(fuel_model(o["distance_km"], o["speed_tw_kn"], o["env"], o["vessel_params"], self.calib))
                for o in observations
            ],
            dtype=float,
        )


class ReferencePhysicsReducedModel:
    """Reduced-feature physics baseline for reference comparison."""

    def fit(self, observations: Sequence[dict[str, Any]]) -> "ReferencePhysicsReducedModel":
        _ = observations
        return self

    def predict(self, observations: Sequence[dict[str, Any]]) -> np.ndarray:
        preds: list[float] = []
        for o in observations:
            vp = o.get("vessel_params", {})
            env = o.get("env", {})
            L = float(vp.get("L", 100.0))
            B = float(vp.get("B", 18.0))
            T = float(vp.get("T", 6.0))

            simple_vessel = {
                "L": L,
                "B": B,
                "T": T,
                "S": float(vp.get("S", 2.2 * (L * B + L * T))),
                "CB": float(vp.get("CB", 0.70)),
                "MCR": float(vp.get("MCR", 10000.0)),
                "sfoc_at_mcr": float(vp.get("sfoc_at_mcr", 180.0)),
                "eta_0": float(vp.get("eta_0", 0.72)),
                "eta_wave_loss": float(vp.get("eta_wave_loss", 0.03)),
                "include_aux": False,
                "include_boiler": False,
                "months_since_cleaning": 0.0,
            }
            simple_env = {
                "wind_kn": float(env.get("wind_kn", 0.0)),
                "wind_angle_deg": float(env.get("wind_angle_deg", 0.0)),
                "Hs": float(env.get("Hs", 0.0)),
                "Tp": float(env.get("Tp", 0.0)),
                "wave_angle_deg": float(env.get("wave_angle_deg", 0.0)),
                "current_kn": float(env.get("current_kn", 0.0)),
                "current_angle_deg": float(env.get("current_angle_deg", 180.0)),
                "sst_c": float(env.get("sst_c", 15.0)),
                "depth_m": float(env.get("depth_m", 0.0)),
            }
            preds.append(float(fuel_model(o["distance_km"], o["speed_tw_kn"], simple_env, simple_vessel)))
        return np.asarray(preds, dtype=float)


@dataclass
class _RidgeRegressor:
    lam: float = 1e-2

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_RidgeRegressor":
        xx = np.asarray(x, dtype=float)
        yy = np.asarray(y, dtype=float)
        mu = xx.mean(axis=0)
        sd = xx.std(axis=0) + 1e-12
        xs = (xx - mu) / sd
        X = np.column_stack([np.ones(len(xs)), xs])
        reg = self.lam * np.eye(X.shape[1])
        reg[0, 0] = 0.0
        self.coef_ = np.linalg.solve(X.T @ X + reg, X.T @ yy)
        self.mu_ = mu
        self.sd_ = sd
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        xx = np.asarray(x, dtype=float)
        xs = (xx - self.mu_) / self.sd_
        X = np.column_stack([np.ones(len(xs)), xs])
        return X @ self.coef_


class _MLPRegressor:
    """2-hidden-layer MLP trained with Adam optimiser, pure numpy — no sklearn needed.

    Architecture: input -> 64 (ReLU) -> 32 (ReLU) -> 1 (linear)
    Inputs are z-score normalised before training and inference.
    """

    def __init__(
        self,
        hidden: tuple[int, ...] = (64, 32),
        lr: float = 1e-3,
        epochs: int = 300,
        batch: int = 64,
        lam: float = 1e-4,
        seed: int = 42,
    ) -> None:
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.batch = batch
        self.lam = lam
        self.seed = seed

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_MLPRegressor":
        rng = np.random.default_rng(self.seed)
        xx = np.asarray(x, dtype=float)
        yy = np.asarray(y, dtype=float).reshape(-1, 1)

        self.mu_ = xx.mean(axis=0)
        self.sd_ = xx.std(axis=0) + 1e-12
        xs = (xx - self.mu_) / self.sd_

        n, d = xs.shape
        sizes = [d, *self.hidden, 1]
        self.W_: list[np.ndarray] = []
        self.b_: list[np.ndarray] = []
        for i in range(len(sizes) - 1):
            # He initialisation for ReLU layers
            self.W_.append(rng.standard_normal((sizes[i], sizes[i + 1])) * np.sqrt(2.0 / sizes[i]))
            self.b_.append(np.zeros((1, sizes[i + 1])))

        # Adam state
        mW = [np.zeros_like(w) for w in self.W_]
        vW = [np.zeros_like(w) for w in self.W_]
        mb = [np.zeros_like(b) for b in self.b_]
        vb = [np.zeros_like(b) for b in self.b_]
        b1, b2, eps = 0.9, 0.999, 1e-8
        t = 0

        idx = np.arange(n)
        for _ in range(self.epochs):
            rng.shuffle(idx)
            for start in range(0, n, self.batch):
                bi = idx[start : start + self.batch]
                xb, yb = xs[bi], yy[bi]

                # Forward pass
                acts: list[np.ndarray] = [xb]
                for i, (W, b) in enumerate(zip(self.W_, self.b_)):
                    z = acts[-1] @ W + b
                    acts.append(np.maximum(z, 0) if i < len(self.W_) - 1 else z)

                # Backward pass
                t += 1
                delta = (acts[-1] - yb) / len(bi)
                for i in range(len(self.W_) - 1, -1, -1):
                    dW = acts[i].T @ delta + self.lam * self.W_[i]
                    db = delta.sum(axis=0, keepdims=True)
                    # Adam
                    mW[i] = b1 * mW[i] + (1 - b1) * dW
                    vW[i] = b2 * vW[i] + (1 - b2) * dW ** 2
                    mb[i] = b1 * mb[i] + (1 - b1) * db
                    vb[i] = b2 * vb[i] + (1 - b2) * db ** 2
                    self.W_[i] -= self.lr * (mW[i] / (1 - b1 ** t)) / (np.sqrt(vW[i] / (1 - b2 ** t)) + eps)
                    self.b_[i] -= self.lr * (mb[i] / (1 - b1 ** t)) / (np.sqrt(vb[i] / (1 - b2 ** t)) + eps)
                    if i > 0:
                        delta = (delta @ self.W_[i].T) * (acts[i] > 0)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        a = (np.asarray(x, dtype=float) - self.mu_) / self.sd_
        for i, (W, b) in enumerate(zip(self.W_, self.b_)):
            z = a @ W + b
            a = np.maximum(z, 0) if i < len(self.W_) - 1 else z
        return a.ravel()


class _PNNRegressor:
    """General Regression Neural Network (PNN for regression, Specht 1991).

    Kernel-weighted sum: f(x) = Σ yᵢ·K(x,xᵢ) / Σ K(x,xᵢ)
    where K(x,xᵢ) = exp(−‖x−xᵢ‖² / (2σ²)).

    Bandwidth σ is selected by leave-one-out cross-validation on a random
    subsample of up to 500 training points.

    Parameters
    ----------
    sigma : float or None
        Fixed bandwidth. None selects σ by LOO-CV.
    cv_bandwidths : int
        Number of σ candidates on a log-spaced grid [0.1, 10].
    seed : int
        RNG seed for subsampling.

    References
    ----------
    Specht, D. F. (1991). A general regression neural network.
    IEEE Transactions on Neural Networks, 2(6), 568–576.
    """

    def __init__(
        self,
        sigma: float | None = None,
        cv_bandwidths: int = 20,
        seed: int = 42,
    ) -> None:
        self.sigma = sigma
        self.cv_bandwidths = cv_bandwidths
        self.seed = seed

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_PNNRegressor":
        xx = np.asarray(x, dtype=float)
        yy = np.asarray(y, dtype=float)

        self.mu_ = xx.mean(axis=0)
        self.sd_ = xx.std(axis=0) + 1e-12
        xs = (xx - self.mu_) / self.sd_

        # Precompute squared norms for efficient distance computation at predict time
        self._xs = xs
        self._y = yy
        self._xs_sq = (xs ** 2).sum(axis=1)  # (n_train,)

        if self.sigma is None:
            rng = np.random.default_rng(self.seed)
            n = len(xs)
            sub_n = min(n, 500)
            idx = rng.choice(n, size=sub_n, replace=False)
            xs_s, y_s = xs[idx], yy[idx]
            xs_s_sq = self._xs_sq[idx]

            # LOO pairwise squared distances via ||a-b||² = ||a||²+||b||²-2aᵀb
            cross = xs_s @ xs_s.T  # (sub_n, sub_n)
            D2 = xs_s_sq[:, None] + xs_s_sq[None, :] - 2.0 * cross
            np.fill_diagonal(D2, np.inf)

            sigmas = np.logspace(-1, 1, self.cv_bandwidths)
            best_sigma, best_mse = sigmas[0], np.inf
            for s in sigmas:
                W = np.exp(-D2 / (2.0 * s ** 2))
                denom = W.sum(axis=1)
                denom = np.where(denom < 1e-300, 1e-300, denom)
                y_hat = (W @ y_s) / denom
                mse = float(np.mean((y_hat - y_s) ** 2))
                if mse < best_mse:
                    best_mse, best_sigma = mse, s

            self.sigma_ = float(best_sigma)
        else:
            self.sigma_ = float(self.sigma)

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        xx = np.asarray(x, dtype=float)
        xs = (xx - self.mu_) / self.sd_
        xs_sq = (xs ** 2).sum(axis=1)  # (n_test,)

        n_test = len(xs)
        preds = np.empty(n_test, dtype=float)
        inv_2s2 = 1.0 / (2.0 * self.sigma_ ** 2)

        # Process in batches to bound memory use (≈ batch × n_train × 8 bytes)
        batch = 256
        for start in range(0, n_test, batch):
            end = min(start + batch, n_test)
            # D2[i,j] = ||xs_test[i] - xs_train[j]||²
            cross = xs[start:end] @ self._xs.T  # (batch, n_train)
            D2 = xs_sq[start:end, None] + self._xs_sq[None, :] - 2.0 * cross
            W = np.exp(-D2 * inv_2s2)  # (batch, n_train)
            denom = W.sum(axis=1)
            denom = np.where(denom < 1e-300, 1e-300, denom)
            preds[start:end] = (W @ self._y) / denom

        return preds


class _SklearnRegressorWrap:
    """Z-score normalising wrapper for sklearn regressors.

    Ensures all sklearn models receive standardised input regardless of
    feature scale, matching the normalisation applied by the internal
    Ridge / MLP / PNN regressors.
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_SklearnRegressorWrap":
        xx = np.asarray(x, dtype=float)
        yy = np.asarray(y, dtype=float)
        self._mu = xx.mean(axis=0)
        self._sd = xx.std(axis=0) + 1e-12
        self._model.fit((xx - self._mu) / self._sd, yy)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        xs = (np.asarray(x, dtype=float) - self._mu) / self._sd
        return np.asarray(self._model.predict(xs), dtype=float)


def _make_learner(name: str, **kwargs: Any) -> Any:
    """Instantiate a residual learner by name.

    All returned objects expose ``.fit(x, y)`` and ``.predict(x)``.

    Parameters
    ----------
    name : str
        Learner backend. One of ``"ridge"``, ``"mlp"``, ``"pnn"``,
        ``"xgboost"``, ``"catboost"``, ``"lightgbm"``, ``"rf"``, ``"gbt"``,
        ``"et"``, ``"dt"``, ``"ada"``, ``"bag"``, ``"lasso"``, ``"elastic"``,
        ``"huber"``, ``"bayesian_ridge"``, ``"knn"``, ``"svr"``, ``"gpr"``.
    **kwargs
        Hyper-parameters forwarded to the learner constructor.

    Returns
    -------
    object
        Learner instance compatible with sklearn ``.fit`` / ``.predict`` protocol.

    Raises
    ------
    ImportError
        If the requested backend is not installed.
    ValueError
        If *name* is not recognised.
    """
    name = name.lower()

    if name == "ridge":
        return _RidgeRegressor(lam=float(kwargs.get("lam", 1e-2)))

    if name == "mlp":
        return _MLPRegressor(
            hidden=tuple(kwargs.get("hidden", (64, 32))),
            lr=float(kwargs.get("lr", 1e-3)),
            epochs=int(kwargs.get("epochs", 300)),
            batch=int(kwargs.get("batch", 64)),
            lam=float(kwargs.get("lam", 1e-4)),
            seed=int(kwargs.get("seed", 42)),
        )

    if name == "xgboost":
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError(
                "xgboost is not installed. Run: pip install xgboost"
            ) from exc

        class _XGBWrap:
            def __init__(self, params: dict[str, Any], n_rounds: int) -> None:
                self._params = params
                self._n_rounds = n_rounds

            def fit(self, x: np.ndarray, y: np.ndarray) -> "_XGBWrap":
                self._model = xgb.train(
                    self._params,
                    xgb.DMatrix(x, label=y),
                    num_boost_round=self._n_rounds,
                    verbose_eval=False,
                )
                return self

            def predict(self, x: np.ndarray) -> np.ndarray:
                return self._model.predict(xgb.DMatrix(x))

        params = {
            "max_depth": int(kwargs.get("max_depth", 4)),
            "eta": float(kwargs.get("learning_rate", 0.05)),
            "subsample": float(kwargs.get("subsample", 0.8)),
            "colsample_bytree": float(kwargs.get("colsample_bytree", 0.8)),
            "seed": int(kwargs.get("random_state", 42)),
            "verbosity": 0,
            "objective": "reg:squarederror",
        }
        return _XGBWrap(params, int(kwargs.get("n_estimators", 300)))

    if name == "catboost":
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise ImportError(
                "catboost is not installed. Run: pip install 'vessel-fuel-model[ml]'"
            ) from exc
        return CatBoostRegressor(
            iterations=int(kwargs.get("iterations", 300)),
            depth=int(kwargs.get("depth", 4)),
            learning_rate=float(kwargs.get("learning_rate", 0.05)),
            random_seed=int(kwargs.get("random_state", 42)),
            verbose=False,
        )

    if name == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError(
                "lightgbm is not installed. Run: pip install lightgbm"
            ) from exc

        class _LGBWrap:
            def __init__(self, params: dict[str, Any], n_rounds: int) -> None:
                self._params = params
                self._n_rounds = n_rounds

            def fit(self, x: np.ndarray, y: np.ndarray) -> "_LGBWrap":
                self._model = lgb.train(
                    self._params,
                    lgb.Dataset(x, label=y),
                    num_boost_round=self._n_rounds,
                )
                return self

            def predict(self, x: np.ndarray) -> np.ndarray:
                return self._model.predict(x)

        params = {
            "max_depth": int(kwargs.get("max_depth", 4)),
            "learning_rate": float(kwargs.get("learning_rate", 0.05)),
            "num_leaves": int(kwargs.get("num_leaves", 31)),
            "subsample": float(kwargs.get("subsample", 0.8)),
            "seed": int(kwargs.get("random_state", 42)),
            "verbose": -1,
            "objective": "regression",
        }
        return _LGBWrap(params, int(kwargs.get("n_estimators", 300)))

    if name == "pnn":
        return _PNNRegressor(
            sigma=kwargs.get("sigma", None),
            cv_bandwidths=int(kwargs.get("cv_bandwidths", 20)),
            seed=int(kwargs.get("seed", 42)),
        )

    if name in ("rf", "random_forest"):
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is not installed. Run: pip install 'vessel-fuel-model[ml]'"
            ) from exc
        return RandomForestRegressor(
            n_estimators=int(kwargs.get("n_estimators", 200)),
            max_depth=kwargs.get("max_depth", None),
            random_state=int(kwargs.get("random_state", 42)),
            n_jobs=int(kwargs.get("n_jobs", -1)),
        )

    if name in ("gbt", "gradient_boosting"):
        try:
            from sklearn.ensemble import GradientBoostingRegressor
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is not installed. Run: pip install 'vessel-fuel-model[ml]'"
            ) from exc
        return GradientBoostingRegressor(
            n_estimators=int(kwargs.get("n_estimators", 200)),
            max_depth=int(kwargs.get("max_depth", 3)),
            learning_rate=float(kwargs.get("learning_rate", 0.1)),
            random_state=int(kwargs.get("random_state", 42)),
        )

    if name in ("et", "extra_trees"):
        try:
            from sklearn.ensemble import ExtraTreesRegressor
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is not installed. Run: pip install 'vessel-fuel-model[ml]'"
            ) from exc
        return ExtraTreesRegressor(
            n_estimators=int(kwargs.get("n_estimators", 300)),
            max_depth=kwargs.get("max_depth", 15),
            random_state=int(kwargs.get("random_state", 42)),
            n_jobs=int(kwargs.get("n_jobs", -1)),
        )

    if name in ("dt", "decision_tree"):
        try:
            from sklearn.tree import DecisionTreeRegressor
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is not installed. Run: pip install 'vessel-fuel-model[ml]'"
            ) from exc
        return DecisionTreeRegressor(
            max_depth=kwargs.get("max_depth", 10),
            random_state=int(kwargs.get("random_state", 42)),
        )

    if name in ("ada", "adaboost"):
        try:
            from sklearn.ensemble import AdaBoostRegressor
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is not installed. Run: pip install 'vessel-fuel-model[ml]'"
            ) from exc
        return AdaBoostRegressor(
            n_estimators=int(kwargs.get("n_estimators", 200)),
            learning_rate=float(kwargs.get("learning_rate", 0.1)),
            random_state=int(kwargs.get("random_state", 42)),
        )

    if name in ("bag", "bagging"):
        try:
            from sklearn.ensemble import BaggingRegressor
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is not installed. Run: pip install 'vessel-fuel-model[ml]'"
            ) from exc
        return BaggingRegressor(
            n_estimators=int(kwargs.get("n_estimators", 100)),
            random_state=int(kwargs.get("random_state", 42)),
            n_jobs=int(kwargs.get("n_jobs", -1)),
        )

    if name == "ols":
        try:
            from sklearn.linear_model import LinearRegression
        except ImportError as exc:
            raise ImportError("scikit-learn is not installed. Run: pip install scikit-learn") from exc
        return _SklearnRegressorWrap(LinearRegression())

    if name in ("ridge_sklearn", "ridge_sk"):
        try:
            from sklearn.linear_model import Ridge
        except ImportError as exc:
            raise ImportError("scikit-learn is not installed. Run: pip install scikit-learn") from exc
        return _SklearnRegressorWrap(Ridge(alpha=float(kwargs.get("alpha", 1.0))))

    if name == "lasso":
        try:
            from sklearn.linear_model import Lasso
        except ImportError as exc:
            raise ImportError("scikit-learn is not installed. Run: pip install scikit-learn") from exc
        return _SklearnRegressorWrap(
            Lasso(alpha=float(kwargs.get("alpha", 1e-3)),
                  max_iter=int(kwargs.get("max_iter", 5000)))
        )

    if name in ("elastic", "elasticnet"):
        try:
            from sklearn.linear_model import ElasticNet
        except ImportError as exc:
            raise ImportError("scikit-learn is not installed. Run: pip install scikit-learn") from exc
        return _SklearnRegressorWrap(
            ElasticNet(alpha=float(kwargs.get("alpha", 1e-3)),
                       l1_ratio=float(kwargs.get("l1_ratio", 0.5)),
                       max_iter=int(kwargs.get("max_iter", 5000)))
        )

    if name == "huber":
        try:
            from sklearn.linear_model import HuberRegressor
        except ImportError as exc:
            raise ImportError("scikit-learn is not installed. Run: pip install scikit-learn") from exc
        return _SklearnRegressorWrap(
            HuberRegressor(epsilon=float(kwargs.get("epsilon", 1.35)),
                           max_iter=int(kwargs.get("max_iter", 500)))
        )

    if name == "bayesian_ridge":
        try:
            from sklearn.linear_model import BayesianRidge
        except ImportError as exc:
            raise ImportError("scikit-learn is not installed. Run: pip install scikit-learn") from exc
        return _SklearnRegressorWrap(
            BayesianRidge(
                max_iter=int(kwargs.get("max_iter", 300)),
                alpha_1=float(kwargs.get("alpha_1", 1e-6)),
                alpha_2=float(kwargs.get("alpha_2", 1e-6)),
                lambda_1=float(kwargs.get("lambda_1", 1e-6)),
                lambda_2=float(kwargs.get("lambda_2", 1e-6)),
            )
        )

    if name in ("knn", "kneighbors"):
        try:
            from sklearn.neighbors import KNeighborsRegressor
        except ImportError as exc:
            raise ImportError("scikit-learn is not installed. Run: pip install scikit-learn") from exc
        return _SklearnRegressorWrap(
            KNeighborsRegressor(
                n_neighbors=int(kwargs.get("n_neighbors", 5)),
                weights=str(kwargs.get("weights", "distance")),
                n_jobs=int(kwargs.get("n_jobs", -1)),
            )
        )

    if name == "svr":
        try:
            from sklearn.svm import SVR
        except ImportError as exc:
            raise ImportError("scikit-learn is not installed. Run: pip install scikit-learn") from exc
        return _SklearnRegressorWrap(
            SVR(
                kernel=str(kwargs.get("kernel", "rbf")),
                C=float(kwargs.get("C", 10.0)),
                epsilon=float(kwargs.get("epsilon", 0.1)),
                gamma=kwargs.get("gamma", "scale"),
            )
        )

    if name in ("quantile", "quantile_regression"):
        try:
            from sklearn.linear_model import QuantileRegressor
        except ImportError as exc:
            raise ImportError("scikit-learn is not installed. Run: pip install scikit-learn") from exc
        return _SklearnRegressorWrap(
            QuantileRegressor(
                quantile=float(kwargs.get("quantile", 0.5)),
                alpha=float(kwargs.get("alpha", 0.0)),
            )
        )

    if name in ("gpr", "gaussian_process"):
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
        except ImportError as exc:
            raise ImportError("scikit-learn is not installed. Run: pip install scikit-learn") from exc
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        return _SklearnRegressorWrap(
            GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=int(kwargs.get("n_restarts_optimizer", 3)),
                random_state=int(kwargs.get("random_state", 42)),
                normalize_y=True,
            )
        )

    # Deep-learning / time-series / symbolic backends (defined later in this module)
    if name == "lstm":
        return LSTMModel(
            hidden_size=int(kwargs.get("hidden_size", 64)),
            num_layers=int(kwargs.get("num_layers", 2)),
            epochs=int(kwargs.get("epochs", 150)),
            lr=float(kwargs.get("lr", 1e-3)),
            dropout=float(kwargs.get("dropout", 0.1)),
            batch_size=int(kwargs.get("batch_size", 64)),
            seed=int(kwargs.get("seed", 42)),
        )

    if name == "prophet":
        return ProphetModel(
            yearly_seasonality=bool(kwargs.get("yearly_seasonality", False)),
            weekly_seasonality=bool(kwargs.get("weekly_seasonality", False)),
            daily_seasonality=bool(kwargs.get("daily_seasonality", False)),
            changepoint_prior_scale=float(kwargs.get("changepoint_prior_scale", 0.05)),
            n_feature_regressors=int(kwargs.get("n_feature_regressors", 5)),
        )

    if name in ("symbolic", "symbolic_regression"):
        return SymbolicRegressionModel(
            generations=int(kwargs.get("generations", 10)),
            population_size=int(kwargs.get("population_size", 1000)),
            random_state=int(kwargs.get("random_state", 42)),
        )

    raise ValueError(
        f"Unknown learner {name!r}. Choose from: "
        "'ridge', 'mlp', 'pnn', 'xgboost', 'catboost', 'lightgbm', 'rf', 'gbt', "
        "'et', 'dt', 'ada', 'bag', "
        "'ols', 'ridge_sklearn', 'lasso', 'elastic', 'huber', 'bayesian_ridge', "
        "'knn', 'svr', 'quantile', 'gpr', "
        "'lstm', 'prophet', 'symbolic'."
    )


class PureMLModel:
    """Pure ML model (no physics term) using engineered feature regression."""

    def __init__(
        self,
        learner: str = "ridge",
        learner_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.reg = _make_learner(learner, **(learner_kwargs or {}))

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PureMLModel":
        self.reg.fit(x, y)
        return self

    def predict_from_matrix(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(self.reg.predict(x), 0.0)


class HybridResidualModel:
    r"""Hybrid model: $\hat{F} = F_{physics} + f_{ML}(x)$.

    The physics term is computed by :class:`PhysicsOnlyModel`; the residual
    correction is learned by the backend selected via *learner*.

    Parameters
    ----------
    learner : str, default ``"ridge"``
        Residual learner backend. One of ``"ridge"``, ``"xgboost"``,
        ``"catboost"``, ``"lightgbm"``, ``"rf"``, ``"gbt"``.
        Tree-based backends require ``pip install 'vessel-fuel-model[ml]'``.
    learner_kwargs : dict, optional
        Hyper-parameters forwarded to the learner constructor (e.g.
        ``{"n_estimators": 500, "max_depth": 6}`` for tree models,
        ``{"lam": 0.05}`` for ridge).
    physics_calib : dict, optional
        Calibration multipliers passed to :class:`PhysicsOnlyModel`.
    """

    def __init__(
        self,
        learner: str = "ridge",
        learner_kwargs: dict[str, Any] | None = None,
        physics_calib: dict[str, float] | None = None,
    ) -> None:
        self.reg = _make_learner(learner, **(learner_kwargs or {}))
        self.physics_model = PhysicsOnlyModel(calib=physics_calib)

    def fit(self, observations: Sequence[dict[str, Any]], x: np.ndarray, y: np.ndarray) -> "HybridResidualModel":
        y_phys = self.physics_model.predict(observations)
        residual = y - y_phys
        self.reg.fit(x, residual)
        return self

    def predict(self, observations: Sequence[dict[str, Any]], x: np.ndarray) -> np.ndarray:
        y_phys = self.physics_model.predict(observations)
        y_res = self.reg.predict(x)
        return np.maximum(y_phys + y_res, 0.0)


class LSTMModel:
    """LSTM neural network for segment-level fuel consumption regression.

    Each sample is treated as a single-timestep input sequence. For proper
    temporal modelling, sort observations by voyage time before calling
    ``fit`` so that sequential structure is preserved in mini-batches.

    Requires ``torch`` (PyTorch). Install with ``pip install torch``.

    Parameters
    ----------
    hidden_size : int
        LSTM hidden state dimension.
    num_layers : int
        Number of stacked LSTM layers.
    epochs : int
        Training epochs.
    lr : float
        Adam learning rate.
    dropout : float
        Dropout between LSTM layers (applied when ``num_layers > 1``).
    batch_size : int
        Mini-batch size.
    seed : int
        Random seed.

    References
    ----------
    Hochreiter, S. and Schmidhuber, J. (1997). Long short-term memory.
    Neural Computation, 9(8), 1735–1780.
    """

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        epochs: int = 150,
        lr: float = 1e-3,
        dropout: float = 0.1,
        batch_size: int = 64,
        seed: int = 42,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.dropout = dropout
        self.batch_size = batch_size
        self.seed = seed
        self._mu: Any = None
        self._sd: Any = None
        self._net: Any = None
        self._torch: Any = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LSTMModel":
        """Fit LSTM to feature matrix.

        Parameters
        ----------
        x : numpy.ndarray, shape (n_samples, n_features)
            Feature matrix from ``build_feature_matrix``.
        y : numpy.ndarray, shape (n_samples,)
            Target fuel consumption (MT).

        Returns
        -------
        LSTMModel
        """
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for LSTMModel. Install with: pip install torch"
            ) from exc

        torch.manual_seed(self.seed)
        xx = np.asarray(x, dtype=float)
        yy = np.asarray(y, dtype=float)

        self._mu = xx.mean(axis=0)
        self._sd = xx.std(axis=0) + 1e-12
        xs = (xx - self._mu) / self._sd

        n, d = xs.shape
        xs_t = torch.tensor(xs, dtype=torch.float32).unsqueeze(1)  # (n, seq=1, d)
        y_t = torch.tensor(yy, dtype=torch.float32).unsqueeze(1)   # (n, 1)

        class _Net(nn.Module):
            def __init__(self, in_size: int, hid: int, layers: int, drop: float) -> None:
                super().__init__()
                self.lstm = nn.LSTM(
                    in_size, hid, layers,
                    batch_first=True,
                    dropout=drop if layers > 1 else 0.0,
                )
                self.head = nn.Linear(hid, 1)

            def forward(self, z: Any) -> Any:
                out, _ = self.lstm(z)
                return self.head(out[:, -1, :])

        self._net = _Net(d, self.hidden_size, self.num_layers, self.dropout)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        rng = np.random.default_rng(self.seed)
        idx = np.arange(n)
        self._net.train()
        for _ in range(self.epochs):
            rng.shuffle(idx)
            for start in range(0, n, self.batch_size):
                bi = torch.tensor(idx[start: start + self.batch_size])
                xb, yb = xs_t[bi], y_t[bi]
                optimizer.zero_grad()
                loss_fn(self._net(xb), yb).backward()
                optimizer.step()

        self._net.eval()
        self._torch = torch
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict fuel consumption.

        Parameters
        ----------
        x : numpy.ndarray, shape (n_samples, n_features)

        Returns
        -------
        numpy.ndarray
        """
        if self._mu is None:
            raise RuntimeError("Call fit() before predict().")
        torch = self._torch
        xs = (np.asarray(x, dtype=float) - self._mu) / self._sd
        xs_t = torch.tensor(xs, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            pred = self._net(xs_t).squeeze(1).numpy()
        return np.maximum(np.asarray(pred, dtype=float), 0.0)


class ProphetModel:
    """Facebook Prophet time-series model for fuel consumption forecasting.

    Each sample is assigned a synthetic hourly timestamp so Prophet can
    model trend. Up to ``n_feature_regressors`` feature columns (after PCA
    reduction to the normalised feature matrix) are added as external
    regressors to improve segment-level accuracy.

    Requires ``prophet``. Install with ``pip install prophet``.

    Parameters
    ----------
    yearly_seasonality : bool
        Include yearly seasonality component.
    weekly_seasonality : bool
        Include weekly seasonality component.
    daily_seasonality : bool
        Include daily seasonality component.
    changepoint_prior_scale : float
        Flexibility of trend changepoints.
    seasonality_prior_scale : float
        Flexibility of seasonality components.
    n_feature_regressors : int or None
        Number of feature columns to add as Prophet regressors. ``None``
        adds all columns.

    References
    ----------
    Taylor, S. J. and Letham, B. (2018). Forecasting at scale.
    The American Statistician, 72(1), 37–45.
    """

    def __init__(
        self,
        yearly_seasonality: bool = False,
        weekly_seasonality: bool = False,
        daily_seasonality: bool = False,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        n_feature_regressors: int | None = 5,
    ) -> None:
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.n_feature_regressors = n_feature_regressors
        self._mu: Any = None
        self._sd: Any = None
        self._n_reg: int = 0
        self._train_n: int = 0
        self._model: Any = None

    def _build_df(self, xs: np.ndarray, y: np.ndarray | None, start_idx: int) -> Any:
        import pandas as pd  # type: ignore[import]
        n = len(xs)
        ds = pd.date_range("2020-01-01", periods=n + start_idx, freq="h")[start_idx:]
        df: dict[str, Any] = {"ds": ds}
        if y is not None:
            df["y"] = y
        for i in range(self._n_reg):
            df[f"reg_{i}"] = xs[:, i]
        return pd.DataFrame(df)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ProphetModel":
        """Fit Prophet model to feature matrix.

        Parameters
        ----------
        x : numpy.ndarray, shape (n_samples, n_features)
        y : numpy.ndarray, shape (n_samples,)

        Returns
        -------
        ProphetModel
        """
        try:
            from prophet import Prophet  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "prophet is not installed. Run: pip install prophet"
            ) from exc
        import logging
        logging.getLogger("prophet").setLevel(logging.WARNING)
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

        xx = np.asarray(x, dtype=float)
        yy = np.asarray(y, dtype=float)
        self._mu = xx.mean(axis=0)
        self._sd = xx.std(axis=0) + 1e-12
        xs = (xx - self._mu) / self._sd

        n_features = xs.shape[1]
        self._n_reg = min(
            n_features,
            self.n_feature_regressors if self.n_feature_regressors is not None else n_features,
        )
        self._train_n = len(xs)

        self._model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
        )
        for i in range(self._n_reg):
            self._model.add_regressor(f"reg_{i}")

        self._model.fit(self._build_df(xs, yy, start_idx=0))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict fuel consumption.

        Parameters
        ----------
        x : numpy.ndarray, shape (n_samples, n_features)

        Returns
        -------
        numpy.ndarray
        """
        if self._mu is None:
            raise RuntimeError("Call fit() before predict().")
        xs = (np.asarray(x, dtype=float) - self._mu) / self._sd
        forecast = self._model.predict(self._build_df(xs, None, start_idx=self._train_n))
        return np.maximum(np.asarray(forecast["yhat"].values, dtype=float), 0.0)


class SymbolicRegressionModel:
    """Symbolic regression to discover interpretable fuel-consumption formulas.

    Tries ``gplearn`` first, then ``pysr``. Raises ``ImportError`` if
    neither package is available.

    Parameters
    ----------
    generations : int
        Number of evolutionary generations.
    population_size : int
        Genetic-programming population size.
    max_samples : float
        Fraction of training samples used per generation.
    function_set : list of str or None
        Primitive functions for ``gplearn``.
        Defaults to ``["add", "sub", "mul", "div", "sqrt", "log", "abs"]``.
    random_state : int
        Random seed.

    References
    ----------
    Schmidt, M. and Lipson, H. (2009). Distilling free-form natural laws
    from experimental data. Science, 324(5923), 81–85.
    """

    def __init__(
        self,
        generations: int = 20,
        population_size: int = 1000,
        max_samples: float = 0.9,
        function_set: list[str] | None = None,
        random_state: int = 42,
    ) -> None:
        self.generations = generations
        self.population_size = population_size
        self.max_samples = max_samples
        self.function_set = function_set or ["add", "sub", "mul", "div", "sqrt", "log", "abs"]
        self.random_state = random_state
        self._mu: Any = None
        self._sd: Any = None
        self._model: Any = None
        self._backend: str = ""

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SymbolicRegressionModel":
        """Fit symbolic regression.

        Parameters
        ----------
        x : numpy.ndarray, shape (n_samples, n_features)
        y : numpy.ndarray, shape (n_samples,)

        Returns
        -------
        SymbolicRegressionModel
        """
        xx = np.asarray(x, dtype=float)
        yy = np.asarray(y, dtype=float)
        self._mu = xx.mean(axis=0)
        self._sd = xx.std(axis=0) + 1e-12
        xs = (xx - self._mu) / self._sd

        try:
            from gplearn.genetic import SymbolicRegressor  # type: ignore[import]
            self._model = SymbolicRegressor(
                generations=self.generations,
                population_size=self.population_size,
                max_samples=self.max_samples,
                function_set=tuple(self.function_set),
                random_state=self.random_state,
                verbose=0,
                n_jobs=-1,
            )
            self._model.fit(xs, yy)
            self._backend = "gplearn"
            return self
        except ImportError:
            pass

        try:
            from pysr import PySRRegressor  # type: ignore[import]
            self._model = PySRRegressor(
                niterations=self.generations,
                population_size=self.population_size,
                unary_operators=["sqrt", "log", "abs"],
                binary_operators=["+", "-", "*", "/"],
                random_state=self.random_state,
                verbosity=0,
            )
            self._model.fit(xs, yy)
            self._backend = "pysr"
            return self
        except ImportError as exc:
            raise ImportError(
                "Neither gplearn nor pysr is installed. "
                "Run: pip install gplearn  or  pip install pysr"
            ) from exc

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict fuel consumption.

        Parameters
        ----------
        x : numpy.ndarray, shape (n_samples, n_features)

        Returns
        -------
        numpy.ndarray
        """
        if self._mu is None:
            raise RuntimeError("Call fit() before predict().")
        xs = (np.asarray(x, dtype=float) - self._mu) / self._sd
        return np.maximum(np.asarray(self._model.predict(xs), dtype=float), 0.0)

    @property
    def best_formula(self) -> str:
        """String representation of the best discovered formula."""
        if self._backend == "gplearn":
            return str(self._model._program)
        if self._backend == "pysr":
            return str(self._model.sympy())
        return "Not fitted"
