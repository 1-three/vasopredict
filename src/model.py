"""
model.py
--------
Implements the full QGWO pipeline:

  DeepQNetwork        — Q-learning agent that adapts GWO's convergence factor
  GreyWolfOptimizer   — base GWO feature-selection algorithm
  QGWO                — hybrid QGWO extending GWO with DQN + 4 enhancements

Then trains the final classifiers (LightGBM, XGBoost, Random Forest,
Gradient Boosting, Logistic Regression) on the optimised feature subset
and compares them against a full-feature baseline.
"""

import time
import random
import collections

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam

np.random.seed(42)
tf.random.set_seed(42)


# =============================================================================
# DEEP Q-NETWORK
# =============================================================================

class DeepQNetwork:
    """
    Deep Q-Network for adaptively controlling GWO's convergence factor.

    State  : [n_features_selected, current_auc, diversity, iteration_progress]
    Actions: 0 = increase exploration  |  1 = decrease exploration  |  2 = maintain
    Reward : improvement in best AUC between iterations
    """

    def __init__(self, state_size: int = 4, action_size: int = 3, learning_rate: float = 0.001):
        self.state_size  = state_size
        self.action_size = action_size

        # Hyperparameters
        self.gamma        = 0.9      # discount factor
        self.epsilon      = 0.9      # initial exploration rate
        self.epsilon_min  = 0.01
        self.epsilon_decay = 0.995
        self.batch_size   = 32

        self.memory = collections.deque(maxlen=2000)
        self.model  = self._build_model()

        print(f"DQN initialised — state_size={state_size}, action_size={action_size}, lr={learning_rate}")

    def _build_model(self):
        inputs  = Input(shape=(self.state_size,))
        x       = Dense(256, activation="relu")(inputs)
        x       = Dropout(0.2)(x)
        x       = Dense(128, activation="relu")(x)
        x       = Dropout(0.2)(x)
        x       = Dense(64,  activation="relu")(x)
        outputs = Dense(self.action_size, activation="linear")(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() <= self.epsilon:
            return np.random.randint(self.action_size)
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return int(np.argmax(q_values[0]))

    def replay(self):
        """Sample a mini-batch and update Q-values via Bellman equation."""
        if len(self.memory) < self.batch_size:
            return

        batch      = random.sample(self.memory, self.batch_size)
        states     = np.array([e[0] for e in batch])
        actions    = np.array([e[1] for e in batch])
        rewards    = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones      = np.array([e[4] for e in batch])

        current_q = self.model.predict(states,      verbose=0)
        next_q    = self.model.predict(next_states, verbose=0)

        for i in range(self.batch_size):
            if dones[i]:
                current_q[i][actions[i]] = rewards[i]
            else:
                current_q[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q[i])

        self.model.fit(states, current_q, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path: str):
        self.model.save_weights(path)

    def load(self, path: str):
        self.model.load_weights(path)


# =============================================================================
# GREY WOLF OPTIMIZER (base)
# =============================================================================

class GreyWolfOptimizer:
    """
    Binary GWO for feature selection.

    Each wolf position is a binary vector [0, 1, ...] where
    1 = feature selected.  Fitness = ROC-AUC on validation set.
    """

    def __init__(self, n_features: int, n_wolves: int = 20, max_iter: int = 50):
        self.n_features = n_features
        self.n_wolves   = n_wolves
        self.max_iter   = max_iter

        # Random initial positions
        self.positions = (np.random.rand(n_wolves, n_features) > 0.5).astype(float)
        self.fitness   = np.zeros(n_wolves)

        # Leadership
        self.alpha_pos,  self.alpha_score = np.zeros(n_features), -np.inf
        self.beta_pos,   self.beta_score  = np.zeros(n_features), -np.inf
        self.delta_pos,  self.delta_score = np.zeros(n_features), -np.inf

        # History
        self.convergence_curve = []
        self.diversity_curve   = []

        print(f"GWO initialised — features={n_features}, wolves={n_wolves}, max_iter={max_iter}")

    def calculate_fitness(self, position, X_train, y_train, X_val, y_val) -> float:
        """
        Train a lightweight Random Forest on the selected feature subset
        and return AUC – complexity_penalty as fitness.
        """
        selected = position > 0.5
        n_sel    = selected.sum()

        if n_sel < 3:
            return 0.0

        complexity_penalty = (n_sel / self.n_features) * 0.1 if n_sel > self.n_features * 0.7 else 0.0

        try:
            clf = RandomForestClassifier(
                n_estimators=50, max_depth=5, random_state=42, n_jobs=-1
            )
            clf.fit(X_train[:, selected], y_train)
            auc = roc_auc_score(y_val, clf.predict_proba(X_val[:, selected])[:, 1])
            return auc - complexity_penalty
        except Exception:
            return 0.0

    def calculate_diversity(self) -> float:
        """Mean standard deviation of wolf positions across features."""
        return float(np.std(self.positions, axis=0).mean())

    def update_leadership(self):
        """Promote top-3 performing wolves to alpha / beta / delta."""
        order = np.argsort(self.fitness)[::-1]

        if self.fitness[order[0]] > self.alpha_score:
            self.alpha_score = self.fitness[order[0]]
            self.alpha_pos   = self.positions[order[0]].copy()

        if self.fitness[order[1]] > self.beta_score:
            self.beta_score = self.fitness[order[1]]
            self.beta_pos   = self.positions[order[1]].copy()

        if self.fitness[order[2]] > self.delta_score:
            self.delta_score = self.fitness[order[2]]
            self.delta_pos   = self.positions[order[2]].copy()


# =============================================================================
# QGWO — Q-Learning Enhanced GWO
# =============================================================================

class QGWO(GreyWolfOptimizer):
    """
    Four key enhancements over standard GWO:
      1. DQN adaptive convergence factor (replaces fixed linear schedule)
      2. Segmented position update  (exploration vs exploitation mode)
      3. Random-jump mechanism      (escape local minima, p = 1%)
      4. Non-dominant replacement   (worst 30% replaced each iteration)
    """

    def __init__(self, n_features: int, n_wolves: int = 20,
                 max_iter: int = 50, dqn: DeepQNetwork = None):
        super().__init__(n_features, n_wolves, max_iter)
        self.dqn = dqn

        # QGWO parameters
        self.beta         = 0.5              # exploration / exploitation threshold
        self.alpha_weights = [0.37, 0.49, 0.28]  # weighted combination for exploitation
        self.r1           = 0.30             # replacement rate
        self.r2           = 0.01             # random jump probability
        self.a            = 2.0              # convergence factor (2 → 0)

        print("QGWO ready with all 4 enhancements.")

    # ------------------------------------------------------------------
    # Helper: state vector for DQN
    # ------------------------------------------------------------------
    def _get_state(self, iteration: int) -> np.ndarray:
        return np.array([
            (self.alpha_pos > 0.5).sum() / self.n_features,
            self.alpha_score,
            self.calculate_diversity(),
            iteration / self.max_iter,
        ])

    # ------------------------------------------------------------------
    # Enhancement 1: DQN-controlled convergence factor
    # ------------------------------------------------------------------
    def _update_a(self, action: int):
        step = 0.1
        if   action == 0: self.a = min(self.a + step, 2.0)
        elif action == 1: self.a = max(self.a - step, 0.0)
        # action == 2: maintain

    # ------------------------------------------------------------------
    # Enhancement 2: Segmented position update
    # ------------------------------------------------------------------
    def _update_position(self, wolf_idx: int) -> np.ndarray:
        if self.a >= self.beta:
            # Exploration: average of leaders + a random omega
            omega_idx = np.random.randint(self.n_wolves)
            new_pos   = (self.alpha_pos + self.beta_pos +
                         self.delta_pos + self.positions[omega_idx]) / 4.0
        else:
            # Exploitation: weighted combination of leaders
            w1, w2, w3 = self.alpha_weights
            new_pos = w1 * self.alpha_pos + w2 * self.beta_pos + w3 * self.delta_pos

        return np.clip(new_pos, 0, 1)

    # ------------------------------------------------------------------
    # Enhancement 3: Random jump
    # ------------------------------------------------------------------
    def _random_jump(self, wolf_idx: int) -> np.ndarray:
        if np.random.random() < self.r2:
            return (np.random.rand(self.n_features) > 0.5).astype(float)
        return self.positions[wolf_idx]

    # ------------------------------------------------------------------
    # Enhancement 4: Replace worst wolves
    # ------------------------------------------------------------------
    def _replace_non_dominant(self):
        n_replace = max(1, int(self.r1 * self.n_wolves))
        for idx in np.argsort(self.fitness)[:n_replace]:
            self.positions[idx] = (np.random.rand(self.n_features) > 0.5).astype(float)

    # ------------------------------------------------------------------
    # Main optimisation loop
    # ------------------------------------------------------------------
    def optimize(self, X_train, y_train, X_val, y_val):
        """
        Run QGWO and return (best_feature_mask, best_auc).
        """
        print("\n" + "=" * 70)
        print("STARTING QGWO OPTIMISATION")
        print("=" * 70)

        for it in range(self.max_iter):
            print(f"\nIteration {it+1}/{self.max_iter}")

            # DQN chooses action
            state  = self._get_state(it)
            action = self.dqn.act(state) if self.dqn else 1  # default: decrease a
            self._update_a(action)
            print(f"  DQN action={action}, a={self.a:.3f}")

            # Evaluate all wolves
            for i in range(self.n_wolves):
                self.positions[i] = self._random_jump(i)
                self.fitness[i]   = self.calculate_fitness(
                    self.positions[i], X_train, y_train, X_val, y_val
                )

            self.update_leadership()

            # Update positions
            for i in range(self.n_wolves):
                self.positions[i] = (self._update_position(i) > 0.5).astype(float)

            self._replace_non_dominant()

            diversity = self.calculate_diversity()
            self.convergence_curve.append(self.alpha_score)
            self.diversity_curve.append(diversity)

            # DQN learns
            if self.dqn and it > 0:
                reward     = self.alpha_score - self.convergence_curve[-2]
                next_state = self._get_state(it + 1)
                done       = (it == self.max_iter - 1)
                self.dqn.remember(state, action, reward, next_state, done)
                self.dqn.replay()

            n_sel = int((self.alpha_pos > 0.5).sum())
            print(f"  Best AUC: {self.alpha_score:.4f} | Features: {n_sel} | Diversity: {diversity:.4f}")

        print("\n" + "=" * 70)
        print(f"OPTIMISATION COMPLETE — Best AUC: {self.alpha_score:.4f}, "
              f"Features: {int((self.alpha_pos > 0.5).sum())}")
        print("=" * 70)
        return self.alpha_pos, self.alpha_score


# =============================================================================
# CLASSIFIER TRAINING
# =============================================================================

MODELS = {
    "Random Forest": lambda pw: RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=20,
        min_samples_leaf=10, random_state=42, n_jobs=-1,
        class_weight="balanced"
    ),
    "XGBoost": lambda pw: xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=pw, random_state=42,
        eval_metric="auc", use_label_encoder=False
    ),
    "LightGBM": lambda pw: lgb.LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced", random_state=42, verbose=-1
    ),
    "Gradient Boosting": lambda pw: GradientBoostingClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, random_state=42
    ),
    "Logistic Regression": lambda pw: LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
}


def train_all_models(X_train_qgwo, y_train, X_val_qgwo, y_val,
                     X_test_qgwo, y_test) -> dict:
    """
    Train all classifiers on QGWO-selected features.

    Returns
    -------
    dict  {model_name: {model, val_auc, test_auc, train_time, y_val_pred, y_test_pred}}
    """
    print("\n" + "=" * 70)
    print("TRAINING FINAL MODELS ON QGWO-SELECTED FEATURES")
    print("=" * 70)

    pos_weight = len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)
    results    = {}

    for name, factory in MODELS.items():
        print(f"\n  Training {name}...")
        clf = factory(pos_weight)

        t0  = time.time()
        clf.fit(X_train_qgwo, y_train)
        elapsed = time.time() - t0

        val_pred  = clf.predict_proba(X_val_qgwo)[:, 1]
        test_pred = clf.predict_proba(X_test_qgwo)[:, 1]
        val_auc   = roc_auc_score(y_val,  val_pred)
        test_auc  = roc_auc_score(y_test, test_pred)

        print(f"    Val AUC: {val_auc:.4f}  |  Test AUC: {test_auc:.4f}  |  Time: {elapsed:.1f}s")

        results[name] = {
            "model":       clf,
            "val_auc":     val_auc,
            "test_auc":    test_auc,
            "train_time":  elapsed,
            "y_val_pred":  val_pred,
            "y_test_pred": test_pred,
        }

    best_name = max(results, key=lambda k: results[k]["test_auc"])
    print(f"\nBest model: {best_name}  (Test AUC = {results[best_name]['test_auc']:.4f})")
    return results, best_name


def baseline_comparison(best_name: str, X_train_sc, y_train,
                        X_val_sc, y_val, X_test_sc, y_test,
                        qgwo_results: dict) -> dict:
    """
    Re-train the best model on the full (unselected) feature set
    and return a comparison DataFrame.
    """
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON: QGWO vs ALL FEATURES")
    print("=" * 70)

    pos_weight = len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)
    baseline   = MODELS[best_name](pos_weight)
    baseline.fit(np.nan_to_num(X_train_sc), y_train)

    base_val_pred  = baseline.predict_proba(np.nan_to_num(X_val_sc))[:, 1]
    base_test_pred = baseline.predict_proba(np.nan_to_num(X_test_sc))[:, 1]
    base_val_auc   = roc_auc_score(y_val,  base_val_pred)
    base_test_auc  = roc_auc_score(y_test, base_test_pred)

    comparison = {
        "baseline_model":     baseline,
        "baseline_val_auc":   base_val_auc,
        "baseline_test_auc":  base_test_auc,
        "baseline_test_pred": base_test_pred,
    }

    improvement = ((qgwo_results[best_name]["test_auc"] - base_test_auc) / base_test_auc) * 100
    print(f"  Baseline Test AUC: {base_test_auc:.4f}")
    print(f"  QGWO    Test AUC:  {qgwo_results[best_name]['test_auc']:.4f}")
    print(f"  Improvement:       {improvement:+.2f}%")

    return comparison


def save_artifacts(best_model, scaler, selected_mask: np.ndarray, best_name: str):
    """Persist model, scaler, and feature mask to models/."""
    import os
    os.makedirs("models", exist_ok=True)

    model_path = f"models/qgwo_{best_name.replace(' ', '_').lower()}.pkl"
    joblib.dump(best_model, model_path)
    joblib.dump(scaler,      "models/feature_scaler.pkl")
    np.save("models/selected_features_mask.npy", selected_mask)

    print(f"\nSaved: {model_path}")
    print("Saved: models/feature_scaler.pkl")
    print("Saved: models/selected_features_mask.npy")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    from preprocessing       import load_raw_data, convert_timestamps, build_dataset
    import feature_engineering as fe

    # Load & preprocess
    data          = load_raw_data()
    data          = convert_timestamps(data)
    lab_w         = fe.create_temporal_windows(data["labs"],   data["cohort"], "charttime", "hadm_id")
    vital_w       = fe.create_temporal_windows(data["vitals"], data["cohort"], "charttime", "stay_id")
    features_df   = fe.create_all_features(lab_w, vital_w, data["cohort"])

    (X, y,
     X_train, X_val, X_test,
     y_train, y_val, y_test,
     X_train_sc, X_val_sc, X_test_sc,
     scaler) = build_dataset(features_df, data["cohort"])

    n_features = X_train_sc.shape[1]

    # QGWO feature selection
    dqn    = DeepQNetwork(state_size=4, action_size=3)
    qgwo   = QGWO(n_features=n_features, n_wolves=20, max_iter=30, dqn=dqn)
    best_features, best_score = qgwo.optimize(
        X_train_sc, y_train, X_val_sc, y_val
    )

    selected_mask = best_features > 0.5
    X_train_q = np.nan_to_num(X_train_sc[:, selected_mask])
    X_val_q   = np.nan_to_num(X_val_sc[:,   selected_mask])
    X_test_q  = np.nan_to_num(X_test_sc[:,  selected_mask])

    # Train classifiers
    results, best_name = train_all_models(
        X_train_q, y_train, X_val_q, y_val, X_test_q, y_test
    )

    baseline = baseline_comparison(
        best_name, X_train_sc, y_train, X_val_sc, y_val,
        X_test_sc, y_test, results
    )

    save_artifacts(results[best_name]["model"], scaler, selected_mask, best_name)
