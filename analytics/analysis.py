
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import matplotlib.pyplot as plt
import seaborn as sns


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "titanic.csv"
CHART_DIR = PROJECT_DIR / "charts"
CHART_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def load_data():
    return pd.read_csv(DATA_PATH)


def classification_workflow(df):
    target = "survived"

    X = df.drop(columns=[target])
    y = df[target]

    # Remove leakage / identifiers not useful for modelling.
    drop_cols = [c for c in ["alive", "class", "deck"] if c in X.columns]
    X = X.drop(columns=drop_cols)

    categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = X.select_dtypes(include=np.number).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric),
        ("cat", categorical_pipeline, categorical)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            max_depth=5
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            oob_score=True
        )
    }

    results = []
    fitted_models = {}

    for name, model in models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]

        results.append({
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "auc": roc_auc_score(y_test, prob)
        })

        fitted_models[name] = pipe

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        PROJECT_DIR / "model_comparison_task10.csv",
        index=False
    )

    # Task 11: class imbalance comparison.
    imbalance_models = {
        "Baseline": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Class Weight Balanced": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )
    }

    imbalance_results = []

    for name, model in imbalance_models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        imbalance_results.append({
            "method": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred)
        })

    smote_pipe = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ])

    smote_pipe.fit(X_train, y_train)
    smote_pred = smote_pipe.predict(X_test)

    imbalance_results.append({
        "method": "SMOTE",
        "accuracy": accuracy_score(y_test, smote_pred),
        "precision": precision_score(y_test, smote_pred),
        "recall": recall_score(y_test, smote_pred),
        "f1": f1_score(y_test, smote_pred)
    })

    pd.DataFrame(imbalance_results).to_csv(
        PROJECT_DIR / "imbalance_comparison_task11.csv",
        index=False
    )

    # Task 12: Random Forest GridSearchCV.
    rf = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        oob_score=True
    )

    rf_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", rf)
    ])

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 5, 10],
        "model__max_features": ["sqrt", "log2"]
    }

    grid = GridSearchCV(
        rf_pipe,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    grid_results = pd.DataFrame(grid.cv_results_)
    grid_results.to_csv(
        PROJECT_DIR / "random_forest_gridsearch_task12.csv",
        index=False
    )

    best_pipeline = grid.best_estimator_

    # Required final pipeline artifact.
    joblib.dump(
        best_pipeline,
        PROJECT_DIR / "best_titanic_pipeline.joblib"
    )

    # Reload test.
    reloaded = joblib.load(
        PROJECT_DIR / "best_titanic_pipeline.joblib"
    )

    sample_prediction = reloaded.predict(X_test.iloc[[0]])
    sample_probability = reloaded.predict_proba(X_test.iloc[[0]])[:, 1]

    print("Best parameters:", grid.best_params_)
    print("Sample prediction:", sample_prediction[0])
    print("Sample survival probability:", sample_probability[0])

    # Decision tree visualisation.
    tree_pipe = fitted_models["Decision Tree"]
    transformed = tree_pipe.named_steps["preprocessor"].transform(X_train)
    tree_model = tree_pipe.named_steps["model"]

    plt.figure(figsize=(18, 10))
    plot_tree(
        tree_model,
        max_depth=3,
        filled=False,
        fontsize=8
    )
    plt.tight_layout()
    plt.savefig(
        CHART_DIR / "06_decision_tree.png",
        dpi=150
    )
    plt.close()

    return results_df


def regression_workflow(df):
    # Fare regression using available numeric predictors.
    work = df.copy()

    features = [c for c in ["age", "pclass", "sibsp", "parch"] if c in work.columns]
    target = "fare"

    work = work[features + [target]].dropna()

    X = work[features]
    y = work[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", LinearRegression())
    ])

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    n = len(y_test)
    p = X_test.shape[1]

    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    residuals = y_test - pred

    plt.figure(figsize=(8, 5))
    plt.scatter(pred, residuals, alpha=0.6)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Fare")
    plt.ylabel("Residual")
    plt.title("Fare Regression Residual Plot")
    plt.tight_layout()
    plt.savefig(
        CHART_DIR / "08_fare_residual_plot.png",
        dpi=150
    )
    plt.close()

    print("Regression MAE:", mae)
    print("Regression RMSE:", rmse)
    print("Regression R2:", r2)
    print("Regression Adjusted R2:", adjusted_r2)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Adjusted_R2": adjusted_r2
    }


if __name__ == "__main__":
    df = load_data()

    print("Dataset shape:", df.shape)

    classification_results = classification_workflow(df)
    print("\nClassification results:")
    print(classification_results)

    regression_results = regression_workflow(df)
    print("\nRegression results:")
    print(regression_results)
