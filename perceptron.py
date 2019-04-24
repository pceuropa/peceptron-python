import numpy as np
import matplotlib.pyplot as plt
from ancillary import plot
from sklearn.metrics import accuracy_score

"""
File: perceptron.py
Author: Rafal Marguzewicz
Email:  rafal.marguzewicz@altimi.com
Github: https://github.com/pceuropa/
Description: Perceptron Rule
"""


class Perceptron(object):

    def __init__(self, eta: float = 0.01, epochs: int = 50):
        self.eta = eta  # learning reate
        self.epochs = epochs

    def train(self, X: np.ndarray, target: np.ndarray):  # Perceptron Rule
        print('Training progress..')
        self.w = np.zeros(1 + X.shape[1])
        self.err = []

        for epoch in range(self.epochs):
            errors = 0
            for x, y in zip(X, target):
                update = self.eta * (y - self.predict(x))
                self.w[1:] += update * x
                self.w[0] += update
                errors += int(update != 0.0)

            accuracy = int(accuracy_score(self.predict(X), target) * 100)
            self.err.append(accuracy)
            plot(X, target, self, epoch, accuracy, self.w)
        return self

    def net_input(self, X: np.ndarray) -> np.float64:
        return np.dot(X, self.w[1:]) + self.w[0]

    def predict(self, X: np.ndarray) -> int:
        return np.where(self.net_input(X) >= 0.0, 1, 0)
