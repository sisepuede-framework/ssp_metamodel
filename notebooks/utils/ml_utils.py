import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate


class PreprocessingUtils:

    @staticmethod
    def pca_reduce(df, var_threshold=0.95):
        pca = PCA(n_components=var_threshold)
        Xp = pca.fit_transform(df)
        return pd.DataFrame(Xp,
                            columns=[f"PC{i+1}" for i in range(Xp.shape[1])],
                            index=df.index), pca
    

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
        param_dist = {
            'n_estimators': [200, 500, 1000, 3000],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.3, 0.5]
        }
        rf = RandomForestRegressor(random_state=self.random_state)
        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
        search = RandomizedSearchCV(
            rf, param_dist, n_iter=n_iter,
            scoring='neg_mean_absolute_error', cv=kf,
            random_state=self.random_state, n_jobs=-1, verbose=1
        )
        search.fit(self.X_train, np.log1p(self.y_train))
        self.best_params = search.best_params_
        print("Best hyperparameters:", self.best_params)

    def train_model(self, log_transform: bool = False):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not preprocessed. Call preprocess() first.")
        model_params = self.best_params or {'n_estimators': 100, 'random_state': self.random_state}
        rf = RandomForestRegressor(**model_params)
        steps = [('scaler', StandardScaler()), ('model', rf)]
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

        # Scatter plot
        plt.figure(figsize=(8, 5))
        plt.scatter(y_pred, self.y_test, alpha=0.6)
        min_val = min(y_pred.min(), self.y_test.min())
        max_val = max(y_pred.max(), self.y_test.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)
        plt.xlabel('Predicted Emissions')
        plt.ylabel('Actual Emissions')
        plt.title('Predicted vs. Actual Emissions')
        plt.tight_layout()
        plt.show()

        # SHAP summary plot
        explainer = shap.Explainer(self.pipeline.named_steps['model'], self.X_test)
        shap_values = explainer(self.X_test)
        shap.summary_plot(shap_values, self.X_test, show=True)

    def cross_validate_model(self, cv_splits: int = 5):
        if self.pipeline is None:
            raise ValueError("Model not trained.")
        scoring = {'MAE': 'neg_mean_absolute_error', 'R2': 'r2', 'RMSE': 'neg_root_mean_squared_error'}
        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
        results = cross_validate(
            self.pipeline, self.X_train, np.log1p(self.y_train) if self._log_transform else self.y_train,
            cv=kf, scoring=scoring, n_jobs=-1, return_train_score=False
        )
        print("\nCross-Validation Results:")
        for metric, scores in results.items():
            if 'test' in metric:
                mean_score = np.mean(-scores) if 'neg' in metric else np.mean(scores)
                print(f"{metric}: Mean={mean_score:.4f}, Std={np.std(scores):.4f}")

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

    def run(self, tune: bool = True, log_transform: bool = False):
        self.preprocess()
        if tune:
            self.tune_hyperparameters()
        self.train_model(log_transform=log_transform)
        self.evaluate_model()
        self.cross_validate_model()
        self.plot_feature_importances()



class GBEmissionsPredictionPipeline:
    def __init__(self, df: pd.DataFrame, target: str = 'total_emissions_last_five_years', test_size: float = 0.2, random_state: int = 42):
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
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

    def tune_hyperparameters(self, n_iter: int = 30, cv_splits: int = 5):
        param_dist = {
            'n_estimators': [100, 200, 500, 1000],
            'learning_rate': [0.005, 0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7, 9, 11],
            'subsample': [0.5, 0.7, 1.0],
            'colsample_bytree': [0.3, 0.5, 0.7, 1.0],
            'min_child_weight': [1, 2, 5, 10],
            'gamma': [0, 0.1, 0.3, 0.5]
        }
        model = xgb.XGBRegressor(random_state=self.random_state, tree_method='hist')
        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring='neg_mean_absolute_error',
            cv=kf,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1
        )
        search.fit(self.X_train, np.log1p(self.y_train))
        self.best_params = search.best_params_
        print("Best hyperparameters:", self.best_params)

    def train_model(self, log_transform: bool = False):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Call preprocess() before train_model().")

        model_params = self.best_params or {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': self.random_state}
        model = xgb.XGBRegressor(**model_params)

        steps = [('scaler', StandardScaler())]
        steps.append(('model', model))

        self.pipeline = Pipeline(steps)
        y = np.log1p(self.y_train) if log_transform else self.y_train
        self.pipeline.fit(self.X_train, y)
        self._log_transform = log_transform

    def evaluate_model(self):
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

        residuals = self.y_test - y_pred
        plt.figure(figsize=(8,5))
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(0, linestyle='--', color='k')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.title('Residuals vs. Predicted')
        plt.show()

        plt.figure(figsize=(8,5))
        plt.scatter(y_pred, self.y_test, alpha=0.6)
        min_val = min(y_pred.min(), self.y_test.min())
        max_val = max(y_pred.max(), self.y_test.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)
        plt.xlabel('Predicted Emissions')
        plt.ylabel('Actual Emissions')
        plt.title('Predicted vs. Actual Emissions')
        plt.tight_layout()
        plt.show()

        explainer = shap.Explainer(self.pipeline.named_steps['model'], self.X_test)
        shap_values = explainer(self.X_test)
        shap.summary_plot(shap_values, self.X_test, show=True)

    def cross_validate_model(self, cv_splits: int = 5):
        if self.pipeline is None:
            raise ValueError("Model not trained.")

        scoring = {'MAE': 'neg_mean_absolute_error', 'R2': 'r2', 'RMSE': 'neg_root_mean_squared_error'}
        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
        results = cross_validate(
            self.pipeline,
            self.X_train,
            np.log1p(self.y_train) if self._log_transform else self.y_train,
            cv=kf,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False
        )

        print("\nCross-Validation Results:")
        for metric, scores in results.items():
            if 'test' in metric:
                mean_score = np.mean(-scores) if 'neg' in metric else np.mean(scores)
                print(f"{metric}: Mean={mean_score:.4f}, Std={np.std(scores):.4f}")

    def plot_feature_importances(self, top_n: int = 10):
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

    def run(self, tune: bool = True, log_transform: bool = False):
        self.preprocess()
        if tune:
            self.tune_hyperparameters()
        self.train_model(log_transform=log_transform)
        self.evaluate_model()
        self.cross_validate_model()
        self.plot_feature_importances()



from sklearn.neural_network import MLPRegressor
from scipy.stats import randint, uniform


class EmissionsPredictionMLP:
    def __init__(self,
                 df: pd.DataFrame,
                 target: str = 'total_emissions_last_five_years',
                 test_size: float = 0.2,
                 random_state: int = 42):
        self.df = df
        self.target = target
        self.test_size = test_size
        self.random_state = random_state

        self.X_train = self.X_test = None
        self.y_train = self.y_test = None

        self.pipeline = None
        self.best_params = None
        self._log_transform = False

    def preprocess(self):
        """Split into train & test sets."""
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state
        )

    def tune_hyperparameters(self,
                              n_iter: int = 20,
                              cv_splits: int = 5):
        """
        Randomized search over MLP hyperparameters, including max_iter
        and solver choice to avoid LBFGS stalls.
        """
        
        param_dist = {
            'hidden_layer_sizes': [(5,), (10,), (20,), (20,10), (50,20), (100,)],
            'activation': ['relu', 'tanh', 'logistic'],
            'alpha': uniform(1e-6, 1e-2),
            'learning_rate_init': uniform(1e-4, 1e-1),
            'batch_size': [32, 64, 128],
            'max_iter': [2000, 5000],
            'solver': ['adam'],
            'early_stopping': [True],
        }

        mlp = MLPRegressor(random_state=self.random_state)
        tscv = TimeSeriesSplit(n_splits=cv_splits)

        search = RandomizedSearchCV(
            estimator=mlp,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring='r2',
            cv=tscv,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1
        )
        # if your target is skewed, use log1p; otherwise y directly
        search.fit(self.X_train, np.log1p(self.y_train))
        self.best_params = search.best_params_
        print("Best hyperparameters:", self.best_params)

    @staticmethod
    def _mlp_importances(estimator):
        """
        Derive a feature‐importance vector from the first layer's weights:
        average absolute weight per input feature.
        """
        coefs = estimator.coefs_[0]   # shape (n_features, n_hidden)
        return np.mean(np.abs(coefs), axis=1)

    def train_model(self,
                    log_transform: bool = False,
                    feature_select: bool = False):
        """
        Build & fit the pipeline: scaling → optional feature‐select → MLP.
        """
        if self.X_train is None:
            raise ValueError("Call preprocess() first.")

        # use tuned params or sensible paper defaults
        params = self.best_params or {
            'hidden_layer_sizes': (5,),
            'activation': 'logistic',
            'solver': 'lbfgs',
            'alpha': 1e-3,
            'learning_rate_init': 1e-3,
            'max_iter': 5000,
            'random_state': self.random_state
        }
        mlp = MLPRegressor(**params)

        steps = [('scaler', StandardScaler())]
        if feature_select:
            steps.append(
                ('feature_select',
                 SelectFromModel(
                     mlp,
                     threshold='median',
                     importance_getter=self._mlp_importances
                 ))
            )
        steps.append(('model', mlp))

        self.pipeline = Pipeline(steps)
        y = np.log1p(self.y_train) if log_transform else self.y_train
        self.pipeline.fit(self.X_train, y)
        self._log_transform = log_transform

    def evaluate_model(self):
        """
        Predict on test set, print MAE/RMSE/R², and residual plot.
        """
        if self.pipeline is None:
            raise ValueError("Train the model before evaluating.")

        y_pred = self.pipeline.predict(self.X_test)
        if self._log_transform:
            y_pred = np.expm1(y_pred)

        mae  = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2   = r2_score(self.y_test, y_pred)

        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²:   {r2:.4f}")

        plt.figure(figsize=(8,5))
        plt.scatter(y_pred, self.y_test - y_pred, alpha=0.6)
        plt.axhline(0, linestyle='--', color='k')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.title('Residuals vs. Predicted (MLP)')
        plt.show()

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

    def run(self,
            tune: bool = False,
            log_transform: bool = False,
            feature_select: bool = False):
        """
        Full workflow: preprocess → (tune) → train → evaluate.
        """
        self.preprocess()
        if tune:
            self.tune_hyperparameters()
        self.train_model(log_transform=log_transform,
                         feature_select=feature_select)
        self.evaluate_model()
