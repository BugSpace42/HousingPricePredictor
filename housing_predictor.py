from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

class HousingPredictor:
    def __init__(self):
        self.load_data()
        self.model = None
        self.metrics = {}

    def load_data(self):
        self.housing = fetch_california_housing()
        self.X = self.housing.data
        self.y = self.housing.target
        self.feature_names = self.housing.feature_names

    def train(self, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42
        )

        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        self.calculate_metrics(y_test, y_pred)
        return y_test, y_pred

    def calculate_metrics(self, y_true, y_pred):
        self.metrics['mae'] = mean_absolute_error(y_true, y_pred)
        self.metrics['mse'] = mean_squared_error(y_true, y_pred)
        self.metrics['rmse'] = np.sqrt(self.metrics['mse'])
        self.metrics['r2'] = r2_score(y_true, y_pred)
