import numpy as np


class SoftmaxRegression:
    def __init__(
            self,
            *,
            penalty="l2",
            alpha=0.0001,
            max_iter=100,
            tol=0.001,
            random_state=None,
            eta0=0.01,
            early_stopping=False,
            validation_fraction=0.1,
            n_iter_no_change=5,
            shuffle=True,
            batch_size=32
    ):
        self._coef = None
        self._intercept = None
        self._n_features = None
        self.best_loss = np.inf
        self.no_improve_count = 0

        self.idx_to_class = None
        self.class_to_idx = None

        self.penalty = penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.eta0 = eta0
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.shuffle = shuffle
        self.batch_size = batch_size

        if self.random_state is not None:
            self.rng = np.random.default_rng(self.random_state)
        else:
            self.rng = np.random.default_rng()

    def get_penalty_grad(self):
        if self._coef is None:
            return 0

        if self.penalty == "l2":
            return 2 * self.alpha * self._coef
        elif self.penalty == "l1":
            return self.alpha * np.sign(self._coef)
        return 0

    def fit(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        classes = sorted(set(y))
        num_classes = len(classes)

        self.class_to_idx = {label: i for i, label in enumerate(classes)}
        self.idx_to_class = {i: label for label, i in self.class_to_idx.items()}

        n_samples, self._n_features = x.shape

        self._coef = np.zeros((self._n_features, num_classes))
        self._intercept = np.zeros(num_classes)

        indices = np.arange(n_samples)

        if self.early_stopping:
            val_size = int(self.validation_fraction * n_samples)
            val_indices = indices[:val_size]
            train_indices = indices[val_size:]
            x_val, y_val = x[val_indices], y[val_indices]
            x, y = x[train_indices], y[train_indices]
            indices = np.arange(len(x))

        for epoch in range(self.max_iter):
            if self.shuffle:
                self.rng.shuffle(indices)

            for start in range(0, len(indices), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                x_batch, y_batch = x[batch_indices], y[batch_indices]

                y_batch_idx = np.array([self.class_to_idx[label] for label in y_batch])
                y_batch_ohe = np.zeros((len(y_batch), num_classes))
                y_batch_ohe[np.arange(len(y_batch)), y_batch_idx] = 1

                probs = self.predict_proba(x_batch)
                grad_w = (x_batch.T @ (probs - y_batch_ohe)) / len(x_batch)
                grad_b = np.mean(probs - y_batch_ohe, axis=0)
                grad_w += self.get_penalty_grad()

                self._coef -= self.eta0 * grad_w
                self._intercept -= self.eta0 * grad_b

                if self.early_stopping:
                    val_probs = self.predict_proba(x_val)
                    y_val_idx = np.array([self.class_to_idx[label] for label in y_val])
                    val_loss = -np.mean(np.log(val_probs[np.arange(len(y_val)), y_val_idx]))
                else:
                    train_probs = self.predict_proba(x)
                    y_train_idx = np.array([self.class_to_idx[label] for label in y])
                    val_loss = -np.mean(np.log(train_probs[np.arange(len(y)), y_train_idx]))

                if self.early_stopping:
                    if val_loss < self.best_loss - self.tol:
                        self.best_loss = val_loss
                        self.no_improve_count = 0
                    else:
                        self.no_improve_count += 1
                        if self.no_improve_count >= self.n_iter_no_change:
                            break

    def predict_proba(self, x):
        x = np.asarray(x)
        logits = np.dot(x, self._coef) + self._intercept
        return self.softmax(logits)

    def predict(self, x):
        probs = self.predict_proba(x)
        class_indices = np.argmax(probs, axis=1)
        return np.array([self.idx_to_class[i] for i in class_indices])

    @staticmethod
    def softmax(z):
        """
        Calculates a softmax normalization over the last axis

        Examples:

        >>> softmax(np.array([1, 2, 3]))
        [0.09003057 0.24472847 0.66524096]

        >>> softmax(np.array([[1, 2, 3], [4, 5, 6]]))
        [[0.09003057 0.24472847 0.66524096]
         [0.03511903 0.70538451 0.25949646]]
        :param z: np.array, size: (d0, d1, ..., dn)
        :return: np.array of the same size as z
        """
        z = np.asarray(z)
        if z.ndim == 1:
            e_z = np.exp(z - np.max(z))
            return e_z / np.sum(e_z)
        else:
            e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
            return e_z / np.sum(e_z, axis=1, keepdims=True)

    @property
    def coef_(self):
        return self._coef

    @property
    def intercept_(self):
        return self._intercept

    @coef_.setter
    def coef_(self, value):
        self._coef = value

    @intercept_.setter
    def intercept_(self, value):
        self._intercept = value
