# housing_predictor_simple.py
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


class HousingPredictor:
    def __init__(self):
        # Загружаем данные
        self.housing = fetch_california_housing()
        self.X = self.housing.data
        self.y = self.housing.target
        self.feature_names = self.housing.feature_names

        self.model = None
        self.metrics = {}
        self.y_test = None
        self.y_pred = None

    def train(self, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42
        )

        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

        self.y_pred = self.model.predict(X_test)
        self.y_test = y_test

        self.metrics['MAE'] = mean_absolute_error(y_test, self.y_pred)
        self.metrics['MSE'] = mean_squared_error(y_test, self.y_pred)
        self.metrics['RMSE'] = np.sqrt(self.metrics['MSE'])
        self.metrics['R2'] = r2_score(y_test, self.y_pred)

        return self.metrics

    def plot_results(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # График реальных vs предсказанных
        ax1.scatter(self.y_test, self.y_pred, alpha=0.5)
        ax1.plot([self.y_test.min(), self.y_test.max()],
                 [self.y_test.min(), self.y_test.max()], 'r--')
        ax1.set_xlabel('Реальные значения (в $100,000)')
        ax1.set_ylabel('Предсказанные значения')
        ax1.set_title('Реальные vs Предсказанные')
        ax1.grid(True)

        errors = self.y_test - self.y_pred
        ax2.hist(errors, bins=30, edgecolor='black')
        ax2.axvline(x=0, color='r', linestyle='--')
        ax2.set_xlabel('Ошибка')
        ax2.set_ylabel('Частота')
        ax2.set_title('Распределение ошибок')
        ax2.grid(True)

        plt.tight_layout()
        plt.show()


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Прогноз стоимости жилья")
        self.root.geometry("600x500")

        self.predictor = HousingPredictor()

        self.setup_ui()

    def setup_ui(self):
        title = tk.Label(self.root, text="Прогнозирование стоимости жилья в Калифорнии",
                         font=("Arial", 14, "bold"))
        title.pack(pady=10)

        info_text = f"Датасет: California Housing\n"
        info_text += f"Образцов: {self.predictor.X.shape[0]}\n"
        info_text += f"Признаков: {self.predictor.X.shape[1]}"
        info_label = tk.Label(self.root, text=info_text, justify=tk.LEFT)
        info_label.pack(pady=5)

        # Настройка test_size
        size_frame = tk.Frame(self.root)
        size_frame.pack(pady=10)

        tk.Label(size_frame, text="Test size:").pack(side=tk.LEFT, padx=5)
        self.test_size = tk.DoubleVar(value=0.2)
        size_spinbox = tk.Spinbox(size_frame, from_=0.1, to=0.5, increment=0.05,
                                  textvariable=self.test_size, width=10)
        size_spinbox.pack(side=tk.LEFT)

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        self.train_btn = tk.Button(button_frame, text="Обучить модель",
                                   command=self.train_model, bg="lightblue", width=15)
        self.train_btn.pack(side=tk.LEFT, padx=5)

        self.plot_btn = tk.Button(button_frame, text="Показать график",
                                  command=self.show_plot, bg="lightgreen", width=15,
                                  state=tk.DISABLED)
        self.plot_btn.pack(side=tk.LEFT, padx=5)

        metrics_frame = tk.LabelFrame(self.root, text="Метрики модели", padx=10, pady=10)
        metrics_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

        self.metrics_text = tk.Text(metrics_frame, height=8, width=50)
        self.metrics_text.pack()

        self.status_label = tk.Label(self.root, text="Готов к работе", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def train_model(self):
        try:
            self.status_label.config(text="Обучение модели...")
            self.root.update()

            test_size = self.test_size.get()
            metrics = self.predictor.train(test_size)

            self.metrics_text.delete(1.0, tk.END)

            output = f"Результаты обучения (test_size={test_size}):\n"
            output += "-" * 40 + "\n"
            output += f"MAE:  {metrics['MAE']:.4f}\n"
            output += f"MSE:  {metrics['MSE']:.4f}\n"
            output += f"RMSE: {metrics['RMSE']:.4f}\n"
            output += f"R²:   {metrics['R2']:.4f}\n"
            output += "-" * 40 + "\n"
            output += f"В среднем ошибка: ${metrics['MAE'] * 100000:.0f}"

            self.metrics_text.insert(1.0, output)

            self.plot_btn.config(state=tk.NORMAL)
            self.status_label.config(text="Модель обучена")

        except Exception as e:
            self.status_label.config(text=f"Ошибка: {str(e)}")

    def show_plot(self):
        self.predictor.plot_results()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()