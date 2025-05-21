import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class EmissionsPredictionPipeline:
    def __init__(self, data_path: str, target: str = 'total_emissions_last_five_years'):
        self.data_path = data_path
        self.target = target
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.pipeline = None

    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        
        # Drop future id and primary id cols
        self.df.drop(columns=['future_id', 'primary_id'], inplace=True)

    def perform_eda(self):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        # Histogram of target
        sns.histplot(self.df[self.target], kde=True)
        plt.title("Distribution of Target Variable")
        plt.show()
        # Correlation with target
        corr = self.df.corr()[self.target].sort_values(ascending=False)
        print("Top 10 Correlated Features:\n", corr.head(11))
        # Heatmap of top features
        top_features = corr.head(11).index
        sns.heatmap(self.df[top_features].corr(), annot=True, fmt=".2f", cmap='coolwarm')
        plt.title("Correlation Heatmap of Top Features")
        plt.show()

    def preprocess(self):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    def train_model(self):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not preprocessed. Call preprocess() first.")
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        self.pipeline.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        if self.pipeline is None or self.X_test is None or self.y_test is None:
            raise ValueError("Model not trained or data not preprocessed.")
        y_pred = self.pipeline.predict(self.X_test)
        print("MAE:", mean_absolute_error(self.y_test, y_pred))
        print("RMSE:", np.sqrt(mean_squared_error(self.y_test, y_pred)))
        print("R^2 Score:", r2_score(self.y_test, y_pred))
        plt.scatter(self.y_test, y_pred, alpha=0.6)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs Predicted Emissions")
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--')
        plt.show()

    # def run(self):
    #     self.load_data()
    #     self.perform_eda()
    #     self.preprocess()
    #     self.train_model()
    #     self.evaluate_model()
