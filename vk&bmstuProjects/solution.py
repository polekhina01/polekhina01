import numpy as np
from sklearn.tree import DecisionTreeRegressor


class GBCustomRegressor:
    def __init__(
            self,
            *,
            learning_rate=0.1,
            n_estimators=100,
            criterion="friedman_mse",
            min_samples_split=2,
            min_samples_leaf=1,
            max_depth=3,
            random_state=None
    ):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.random_state = random_state

        self._initial_prediction = None
        self._trees = []

    def fit(self, x, y):
        x = np.array(x)
        y = np.array(y)

        self._initial_prediction = np.mean(y)
        y_pred = np.full_like(y, fill_value=self._initial_prediction, dtype=float)

        for i in range(self.n_estimators):
            residuals = y - y_pred

            tree = DecisionTreeRegressor(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state
            )

            tree.fit(x, residuals)

            update = tree.predict(x)
            y_pred += self.learning_rate * update

            self._trees.append(tree)

    def predict(self, x):
        x = np.array(x)
        y_pred = np.full(shape=(x.shape[0],), fill_value=self._initial_prediction, dtype=float)

        for tree in self._trees:
            y_pred += self.learning_rate * tree.predict(x)

        return y_pred

    @property
    def estimators_(self):
        return self._trees


class GBCustomClassifier:
    def __init__(
            self,
            *,
            learning_rate=0.1,
            n_estimators=100,
            criterion="friedman_mse",
            min_samples_split=2,
            min_samples_leaf=1,
            max_depth=3,
            random_state=None
    ):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.random_state = random_state

        self._initial_logit = None
        self._trees = []

    def fit(self, x, y):
        x = np.array(x)
        y = np.array(y).astype(float)

        eps = 1e-5
        p = np.clip(np.mean(y), eps, 1 - eps)
        self._initial_logit = np.log(p / (1 - p))

        logits = np.full_like(y, fill_value=self._initial_logit, dtype=float)

        for i in range(self.n_estimators):
            probas = 1 / (1 + np.exp(-logits))

            residuals = y - probas

            tree = DecisionTreeRegressor(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state
            )

            tree.fit(x, residuals)

            logits += self.learning_rate * tree.predict(x)

            self._trees.append(tree)

    def predict_proba(self, x):
        x = np.array(x)

        logits = np.full(shape=(x.shape[0],), fill_value=self._initial_logit, dtype=float)

        for tree in self._trees:
            logits += self.learning_rate * tree.predict(x)

        probas = 1 / (1 + np.exp(-logits))

        return np.vstack([1 - probas, probas]).T

    def predict(self, x):
        proba = self.predict_proba(x)[:, 1]
        return (proba > 0.5).astype(int)

    @property
    def estimators_(self):
        return self._trees
