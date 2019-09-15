"""Simple implementation of Linear Regression."""
import numpy as np

import cross_validation


class LinRegressor:
    """Simple algorithm to fit a linear regression model."""

    def __init__(self):
        self.reg_coeff = None  # type: np.ndarray
        self.intercept = None  # type: np.ndarray

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray):
        """Root mean squared error."""
        return np.sqrt(np.sum(np.square(y_true - y_pred)) / y_true.size)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinRegressor":
        """Simple linear regression."""
        _num_inst = X.size if X.ndim == 1 else X.shape[0]

        if _num_inst != y.size:
            raise ValueError("Number of instances (got {}) and 'y' "
                             "size (got {}) don't match!".format(
                                 _num_inst, y.size))

        x_mean = X.mean()
        y_mean = y.mean()

        _aux = X - x_mean

        self.reg_coeff = np.dot(_aux, y - y_mean) / np.dot(_aux, _aux)

        self.intercept = y_mean - self.reg_coeff * x_mean

        return self

    def predict(self, vals: np.ndarray) -> np.ndarray:
        """Predict the fitted function values for ``vals``.

        Let a be the intercept coefficients and b the regression coefficiets.
        Then, this methods simply calculates y_{i} = f(vals_{i}) as follows:

            y_{i} = a + b * vals_{i}

        Arguments
        ---------
        vals : :obj:`np.ndarray`
            Points to evaluate model function.

        Returns
        -------
        :obj:`np.ndarray`
            Array such that every entry is y_{i} = f(vals_{i}).
        """
        return vals * self.reg_coeff + self.intercept


def _test():
    import matplotlib.pyplot as plt
    random_state = 16

    np.random.seed(random_state)
    pop_size = 30
    num_folds = 9

    X = np.arange(pop_size) + np.random.normal(
        loc=0.0, scale=0.01, size=pop_size)
    y = np.arange(pop_size) + np.random.normal(
        loc=0.0, scale=1.0, size=pop_size)

    pop = np.hstack((X.reshape(-1, 1), y.reshape(-1, 1)))

    errors = np.zeros(num_folds)

    for fold_id, fold in enumerate(
            cross_validation.kfold_cv(
                pop, k=num_folds, random_state=random_state)):
        data_test, data_train = fold

        X_test, y_test = data_test[:, 0], data_test[:, 1]
        X_train, y_train = data_train[:, 0], data_train[:, 1]

        model = LinRegressor().fit(X_train, y_train)
        y_pred = model.predict(X_test)

        errors[fold_id] = model.rmse(y_test, y_pred)

        plt.subplot(3, 3, 1 + fold_id)
        plt.plot(X, y, label="True data")
        plt.plot(
            X_test, y_pred, 'o', label="RMSE: {:.2f}".format(errors[fold_id]))
        plt.legend()
        plt.title(str(fold_id))

    total_error = LinRegressor.rmse(errors, np.zeros(num_folds))
    print("Total RMSE:", total_error)

    plt.show()


if __name__ == "__main__":
    _test()
