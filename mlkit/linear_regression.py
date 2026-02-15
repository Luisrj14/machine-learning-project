import numpy as np

class LinearRegression:
    def __init__(self):
        self.m = None  # slope
        self.b = None  # intercept

    def fit(self, x, y):
        x = np.array(x)
        y = np.array(y)

        x_mean = np.mean(x)
        y_mean = np.mean(y)

        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        self.m = numerator / denominator
        self.b = y_mean - self.m * x_mean

    def predict(self, x):
        if self.m is None or self.b is None:
            raise ValueError("Model has not been trained yet!")

        x = np.array(x)
        return self.m * x + self.b
