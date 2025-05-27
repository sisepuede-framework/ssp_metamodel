import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor


class RFEmissionsPredictionPipeline:
    def __init__(self, df, target: str = 'total_emissions_last_five_years', test_size: float = 0.2, random_state: int = 42):
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.df = df
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.pipeline = None
        self.best_params = None

    def preprocess(self):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

    def tune_hyperparameters(self, n_iter: int = 30, cv_splits: int = 5):
        """
        Performs RandomizedSearchCV on a RandomForest to find best hyperparameters.
        """
        param_dist = {
            'n_estimators': [200, 500, 1000],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.3, 0.5]
        }
        rf = RandomForestRegressor(random_state=self.random_state)
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        search = RandomizedSearchCV(
            rf, param_dist, n_iter=n_iter,
            scoring='neg_mean_absolute_error', cv=tscv,
            random_state=self.random_state, n_jobs=-1, verbose=1
        )
        search.fit(self.X_train, np.log1p(self.y_train))
        self.best_params = search.best_params_
        print("Best hyperparameters:", self.best_params)

    def train_model(self, log_transform: bool = False, feature_select: bool = False):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not preprocessed. Call preprocess() first.")
        # Use tuned params or defaults
        model_params = self.best_params or {'n_estimators': 100, 'random_state': self.random_state}
        rf = RandomForestRegressor(**model_params)
        steps = [('scaler', StandardScaler())]
        if feature_select:
            steps.append(('feature_select', SelectFromModel(rf, threshold='median')))
        steps.append(('model', rf))
        self.pipeline = Pipeline(steps)
        y = np.log1p(self.y_train) if log_transform else self.y_train
        self.pipeline.fit(self.X_train, y)
        self._log_transform = log_transform

    def evaluate_model(self):
        if self.pipeline is None or self.X_test is None or self.y_test is None:
            raise ValueError("Model not trained or data not preprocessed.")
        y_pred = self.pipeline.predict(self.X_test)
        if self._log_transform:
            y_pred = np.expm1(y_pred)
        print("MAE:", mean_absolute_error(self.y_test, y_pred))
        print("RMSE:", np.sqrt(mean_squared_error(self.y_test, y_pred)))
        print("R^2 Score:", r2_score(self.y_test, y_pred))
        # Residual plot
        res = self.y_test - y_pred
        plt.figure(figsize=(8,5))
        plt.scatter(y_pred, res, alpha=0.6)
        plt.axhline(0, color='k', linestyle='--')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Predicted')
        plt.show()

        plt.figure(figsize=(8, 5))

        # Scatter: Predicted on X, Actual on Y
        plt.scatter(y_pred, self.y_test, alpha=0.6)

        # Identity line over the combined range
        min_val = min(y_pred.min(), self.y_test.min())
        max_val = max(y_pred.max(), self.y_test.max())
        plt.plot([min_val, max_val],
                [min_val, max_val],
                'k--',
                linewidth=2)

        plt.xlabel('Predicted Emissions')
        plt.ylabel('Actual Emissions')
        plt.title('Predicted vs. Actual Emissions')
        plt.tight_layout()
        plt.show()


    def plot_feature_importances(self, top_n: int = 10):
        if not hasattr(self.pipeline.named_steps['model'], 'feature_importances_'):
            raise ValueError("Model does not have feature_importances_.")
        importances = self.pipeline.named_steps['model'].feature_importances_
        features = self.X_train.columns
        idx = np.argsort(importances)[-top_n:][::-1]
        plt.figure(figsize=(8,5))
        plt.barh(features[idx][::-1], importances[idx][::-1])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.show()

    def run(self, tune: bool = True, log_transform: bool = False, feature_select: bool = False):
        # self.load_data()
        self.preprocess()
        if tune:
            self.tune_hyperparameters()
        self.train_model(log_transform=log_transform, feature_select=feature_select)
        self.evaluate_model()
        self.plot_feature_importances()




class BoostingEmissionsPredictionPipeline:
    def __init__(self,
                 df: pd.DataFrame,
                 target: str = 'total_emissions_last_five_years',
                 test_size: float = 0.2,
                 random_state: int = 42):
        self.df = df
        self.target = target
        self.test_size = test_size
        self.random_state = random_state

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.pipeline = None
        self.best_params = None
        self._log_transform = False

    def preprocess(self):
        """Split df into train/test X and y."""
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state
        )
        print("Train/test split done!")

    def tune_hyperparameters(self,
                              n_iter: int = 30,
                              cv_splits: int = 5):
        """
        Randomized search over GradientBoosting hyperparameters
        using TimeSeriesSplit.
        """
        print("Doing hyperparameter tunning...")
        param_dist = {
                # More & larger trees
                'n_estimators':    [100, 200, 500, 1000, 1500, 2000],

                # Slower learning rates to allow more trees to fit subtleties
                'learning_rate':   [0.005, 0.01, 0.05, 0.1, 0.2],

                # Deeper trees
                'max_depth':       [3, 5, 7, 9, 11, 13],

                # Allow smaller subsamples (more variance) as well as full-sample
                'subsample':       [0.5, 0.6, 0.8, 1.0],

                # Permit very small leaf and split sizes
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf':  [1, 2, 4, 10],

                # Try using all features or standard heuristics
                'max_features':    ['sqrt', 'log2', 0.3, 0.5, 1.0],

                # Different loss functions—“huber” can be more robust to outliers
                'loss':   ['quantile', 'squared_error', 'absolute_error', 'huber'],

                # Control number of leaf nodes to let trees grow larger
                'max_leaf_nodes':  [None, 10, 20, 30, 50]
            }

        gb = GradientBoostingRegressor(random_state=self.random_state)
        tscv = TimeSeriesSplit(n_splits=cv_splits)

        search = RandomizedSearchCV(
            estimator=gb,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring='neg_mean_absolute_error',
            cv=tscv,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1
        )
        # log-transform target for search stability
        search.fit(self.X_train, np.log1p(self.y_train))
        self.best_params = search.best_params_
        print("Best hyperparameters:", self.best_params)

    def train_model(self,
                    log_transform: bool = False,
                    feature_select: bool = False):
        """
        Build a pipeline with scaling, optional feature selection,
        and a GradientBoostingRegressor.
        """

        print("Train model...")
        if self.X_train is None or self.y_train is None:
            raise ValueError("Call preprocess() before train_model().")

        # use tuned params or sensible defaults
        model_params = self.best_params or {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'random_state': self.random_state
        }
        gb = GradientBoostingRegressor(**model_params)

        steps = [
            ('scaler', StandardScaler())
        ]
        if feature_select:
            steps.append(
                ('feature_select', SelectFromModel(gb, threshold='median'))
            )
        steps.append(('model', gb))

        self.pipeline = Pipeline(steps)
        y = np.log1p(self.y_train) if log_transform else self.y_train
        self.pipeline.fit(self.X_train, y)
        self._log_transform = log_transform

    def evaluate_model(self):
        """
        Predict on test set, compute MAE, RMSE, R^2,
        and plot residuals.
        """

        print("Evaluating model...")
        if self.pipeline is None:
            raise ValueError("Train the model before evaluation.")

        y_pred = self.pipeline.predict(self.X_test)
        if self._log_transform:
            y_pred = np.expm1(y_pred)

        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2  = r2_score(self.y_test, y_pred)

        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²:   {r2:.4f}")

        # Residual plot
        residuals = self.y_test - y_pred
        plt.figure(figsize=(8,5))
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(0, linestyle='--', color='k')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.title('Residuals vs. Predicted')
        plt.show()

        plt.figure(figsize=(8, 5))

        # Scatter: Predicted on X, Actual on Y
        plt.scatter(y_pred, self.y_test, alpha=0.6)

        # Identity line over the combined range
        min_val = min(y_pred.min(), self.y_test.min())
        max_val = max(y_pred.max(), self.y_test.max())
        plt.plot([min_val, max_val],
                [min_val, max_val],
                'k--',
                linewidth=2)

        plt.xlabel('Predicted Emissions')
        plt.ylabel('Actual Emissions')
        plt.title('Predicted vs. Actual Emissions')
        plt.tight_layout()
        plt.show()


    def plot_feature_importances(self, top_n: int = 10):
        """
        Bar plot of the top_n feature importances.
        """
        model = self.pipeline.named_steps['model']
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("Model has no feature_importances_ attribute.")

        importances = model.feature_importances_
        features = self.X_train.columns
        idx = np.argsort(importances)[-top_n:][::-1]

        plt.figure(figsize=(8,5))
        plt.barh(features[idx][::-1], importances[idx][::-1])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.show()

    def run(self,
            tune: bool = True,
            log_transform: bool = False,
            feature_select: bool = False):
        """
        Full workflow: preprocess → (tune) → train → evaluate → plot importances
        """
        self.preprocess()
        if tune:
            self.tune_hyperparameters()
        self.train_model(log_transform=log_transform,
                         feature_select=feature_select)
        self.evaluate_model()
        self.plot_feature_importances()
