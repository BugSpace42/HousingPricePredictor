import tkinter as tk
from tkinter import ttk
from housing_predictor import HousingPredictor


class HousingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Прогноз стоимости жилья")

        self.predictor = HousingPredictor()

        self.create_widgets()

    def create_widgets(self):
        self.train_btn = ttk.Button(text="Обучить модель",
                                    command=self.train_model)
        self.train_btn.pack(pady=10)

        self.output = tk.Text(height=10, width=50)
        self.output.pack(pady=10)

    def train_model(self):
        y_test, y_pred = self.predictor.train()

        self.output.delete(1.0, tk.END)
        self.output.insert(1.0, f"MAE: {self.predictor.metrics['mae']:.4f}\n")
        self.output.insert(1.0, f"MSE: {self.predictor.metrics['mse']:.4f}\n")
        self.output.insert(1.0, f"RMSE: {self.predictor.metrics['rmse']:.4f}\n")
        self.output.insert(1.0, f"R²: {self.predictor.metrics['r2']:.4f}\n")
