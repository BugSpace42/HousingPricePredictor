from sklearn.datasets import fetch_california_housing
import pandas as pd

housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target, name='MedHouseVal')

print(f"Размер данных: {X.shape}")
print(f"Признаки: {list(X.columns)}")
print(f"Первые 5 строк:\n{X.head()}")