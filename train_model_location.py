import warnings
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

CSV_COL_X = "x"
CSV_COL_Y = "y"
CSV_COL_FC12 = "FC1,2"
CSV_COL_FC21 = "FC2,1"
CSV_COL_FC22 = "FC2,2"
CSV_COL_FC23 = "FC2,3"
CSV_COL_FC32 = "FC3,2"
FEAT_FC21_DIV_FC22 = "FC2,1/FC2,2"
FEAT_FC23_DIV_FC22 = "FC2,3/FC2,2"
FEAT_FC12_DIV_FC22 = "FC1,2/FC2,2"
FEAT_FC32_DIV_FC22 = "FC3,2/FC2,2"

class GradientBoostingPitCorrosionPredictor:
    def __init__(self):
        self.model = None
        self.features = None
        self.targets = None
        self.training_data = None

    def load_and_preprocess_data(self, file_path):
        self.training_data = pd.read_csv(file_path)

        self.training_data[FEAT_FC21_DIV_FC22] = (
            self.training_data[CSV_COL_FC21] / self.training_data[CSV_COL_FC22]
        )
        self.training_data[FEAT_FC23_DIV_FC22] = (
            self.training_data[CSV_COL_FC23] / self.training_data[CSV_COL_FC22]
        )
        self.training_data[FEAT_FC12_DIV_FC22] = (
            self.training_data[CSV_COL_FC12] / self.training_data[CSV_COL_FC22]
        )
        self.training_data[FEAT_FC32_DIV_FC22] = (
            self.training_data[CSV_COL_FC32] / self.training_data[CSV_COL_FC22]
        )
        self.features = [
            FEAT_FC21_DIV_FC22,
            FEAT_FC23_DIV_FC22,
            FEAT_FC12_DIV_FC22,
            FEAT_FC32_DIV_FC22,
        ]
        self.targets = [CSV_COL_X, CSV_COL_Y]
        X = self.training_data[self.features]
        y = self.training_data[self.targets]
        print("Data loaded and preprocessed successfully")

        return X, y

    def split_data(self, X, y, test_size=0.2, random_state=10):
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def _make_search_pipeline(self):
        base_model = MultiOutputRegressor(GradientBoostingRegressor(random_state=42))
        return Pipeline([("scaler", StandardScaler()), ("model", base_model)])

    def optimize_parameters(self, X_train, y_train, n_iter=50, cv=5):
        """Bayesian search for gradient boosting hyperparameters."""
        print("\nStarting Bayesian hyperparameter optimization...")
        pipeline = self._make_search_pipeline()
        search_space = {
            "model__estimator__n_estimators": Integer(50, 300),
            "model__estimator__learning_rate": Real(0.01, 0.3, prior="log-uniform"),
            "model__estimator__max_depth": Integer(2, 10),
            "model__estimator__min_samples_split": Integer(2, 20),
        }
        bayes_search = BayesSearchCV(
            pipeline,
            search_space,
            n_iter=n_iter,
            cv=cv,
            scoring="r2",
            n_jobs=-1,
            verbose=0,
            random_state=42,
        )
        bayes_search.fit(X_train, y_train)
        print(f"Best parameters: {bayes_search.best_params_}")
        print(f"Best cross-validation R²: {bayes_search.best_score_:.4f}")

        best_estimator = bayes_search.best_estimator_
        best_estimator.best_params_ = bayes_search.best_params_
        best_estimator.best_score_ = bayes_search.best_score_
        return best_estimator

    def train_model(self, X_train, y_train, optimize=True):
        print("\nStarting gradient boosting model training...")
        if optimize:
            self.model = self.optimize_parameters(X_train, y_train)
        else:
            base_model = MultiOutputRegressor(
                GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    subsample=0.9,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                )
            )
            self.model = Pipeline(
                [("scaler", StandardScaler()), ("model", base_model)]
            )
            self.model.fit(X_train, y_train)

        print("Model training completed")
        return self.model

    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model on the test set."""
        print("\nEvaluating model performance...")
        y_pred = self.model.predict(X_test)
        x_col, y_col = self.targets[0], self.targets[1]

        mse_x = mean_squared_error(y_test[x_col], y_pred[:, 0])
        rmse_x = np.sqrt(mse_x)
        mae_x = mean_absolute_error(y_test[x_col], y_pred[:, 0])
        r2_x = r2_score(y_test[x_col], y_pred[:, 0])

        mse_y = mean_squared_error(y_test[y_col], y_pred[:, 1])
        rmse_y = np.sqrt(mse_y)
        mae_y = mean_absolute_error(y_test[y_col], y_pred[:, 1])
        r2_y = r2_score(y_test[y_col], y_pred[:, 1])

        mse_overall = mean_squared_error(y_test, y_pred)
        rmse_overall = np.sqrt(mse_overall)
        mae_overall = mean_absolute_error(y_test, y_pred)
        r2_overall = r2_score(y_test, y_pred)

        residuals_x = y_test[x_col] - y_pred[:, 0]
        residuals_y = y_test[y_col] - y_pred[:, 1]

        results = {
            "X_R2": r2_x,
            "X_RMSE": rmse_x,
            "X_MSE": mse_x,
            "X_MAE": mae_x,
            "X_residual_mean": residuals_x.mean(),
            "X_residual_std": residuals_x.std(),
            "Y_R2": r2_y,
            "Y_RMSE": rmse_y,
            "Y_MSE": mse_y,
            "Y_MAE": mae_y,
            "Y_residual_mean": residuals_y.mean(),
            "Y_residual_std": residuals_y.std(),
            "Overall_R2": r2_overall,
            "Overall_RMSE": rmse_overall,
            "Overall_MSE": mse_overall,
            "Overall_MAE": mae_overall,
        }

        print("=== Gradient Boosting Model Evaluation Results ===")
        print(f"X: R²={r2_x:.4f}, RMSE={rmse_x:.4f}, MAE={mae_x:.4f}")
        print(f"Y: R²={r2_y:.4f}, RMSE={rmse_y:.4f}, MAE={mae_y:.4f}")
        print(f"Overall: R²={r2_overall:.4f}, RMSE={rmse_overall:.4f}, MAE={mae_overall:.4f}")

        return results, y_pred

    def plot_actual_vs_predicted(
        self,
        y_test,
        y_pred,
        save_path="actual_vs_predicted_location.png",
    ):
        """Single figure: X (left) and Y (right) actual vs predicted scatter plots."""
        x_col, y_col = self.targets[0], self.targets[1]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, col, pred_i, coord, color in zip(
            axes,
            (x_col, y_col),
            (0, 1),
            ("X", "Y"),
            ("blue", "green"),
        ):
            actual = y_test[col]
            pred = y_pred[:, pred_i]
            ax.scatter(actual, pred, alpha=0.6, s=25, color=color)
            lo, hi = actual.min(), actual.max()
            ax.plot([lo, hi], [lo, hi], "r--", lw=2, label="Ideal fit")
            ax.set_xlabel(f"Actual {coord} coordinate")
            ax.set_ylabel(f"Predicted {coord} coordinate")
            ax.set_title(f"Gradient Boosting — {coord} coordinate")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="lower right")
            r2 = r2_score(actual, pred)
            rmse = np.sqrt(mean_squared_error(actual, pred))
            ax.text(
                0.05,
                0.95,
                f"R² = {r2:.3g}\nRMSE = {rmse:.3f}",
                transform=ax.transAxes,
                va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
            )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"Saved plot: {save_path}")

    def save_model(self, model_name="gradient_boosting_model"):
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        print(f"\nSaving model: {model_name}")
        joblib.dump(self.model, f"{model_name}.joblib")
        joblib.dump(self.features, "gradient_boosting_features.joblib")
        joblib.dump(self.targets, "gradient_boosting_targets.joblib")
        print("Model and features saved")

def main():
    print("=== Gradient Boosting Pit Location Model Training ===")
    predictor = GradientBoostingPitCorrosionPredictor()
    X, y = predictor.load_and_preprocess_data("data\\Location_data.csv")
    X_train, X_test, y_train, y_test = predictor.split_data(X, y)

    predictor.train_model(X_train, y_train, optimize=True)
    _, y_pred = predictor.evaluate_model(X_test, y_test)
    predictor.plot_actual_vs_predicted(y_test, y_pred)
    predictor.save_model()

    print("\n=== Training Completed ===")


if __name__ == "__main__":
    main()
