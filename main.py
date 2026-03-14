from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

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

housing = fetch_california_housing()
X, y = housing.data, housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# График рассеяния
axes[0].scatter(y_test, y_pred, alpha=0.5)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[0].set_xlabel('Реальные значения')
axes[0].set_ylabel('Предсказания')
axes[0].set_title('Реальные vs Предсказанные')

# Гистограмма ошибок
errors = y_test - y_pred
axes[1].hist(errors, bins=30)
axes[1].set_xlabel('Ошибка')
axes[1].set_ylabel('Частота')
axes[1].set_title('Распределение ошибок')

plt.tight_layout()
plt.show()