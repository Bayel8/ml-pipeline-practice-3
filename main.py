
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# 1. Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Titanic-Dataset.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# 2. Load dataset
df = pd.read_csv(DATA_PATH)

print("DATASET INFORMATION")
print("Dataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

# 3. Select features and target
numeric_features = ["Age", "Fare", "SibSp", "Parch"]
mode_features = ["Pclass"]
unknown_features = ["Sex", "Embarked"]
target = "Survived"

X = df[numeric_features + mode_features + unknown_features]
y = df[target]

# 4. Split data BEFORE preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nDATA SPLIT")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# 5. Numeric pipeline
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 6. Pclass pipeline
pclass_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

# 7. Sex and Embarked pipeline
unknown_pipeline = Pipeline([
    ("imputer", SimpleImputer(
        strategy="constant",
        fill_value="Unknown"
    )),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

# 8. Combine transformations
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("pclass", pclass_pipeline, mode_features),
    ("categorical", unknown_pipeline, unknown_features)
])

# 9. Fit on train only
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test using the fitted preprocessor
X_test_processed = preprocessor.transform(X_test)

# 10. Validate the pipeline
print("\nPIPELINE VALIDATION")
print("Original train shape:", X_train.shape)
print("Original test shape:", X_test.shape)
print("Processed train shape:", X_train_processed.shape)
print("Processed test shape:", X_test_processed.shape)

same_features = (
    X_train_processed.shape[1] == X_test_processed.shape[1]
)

print("Same number of features:", same_features)
print(
    "No missing values in train:",
    not pd.isna(X_train_processed).any()
)
print(
    "No missing values in test:",
    not pd.isna(X_test_processed).any()
)

# 11. Save processed datasets
feature_names = preprocessor.get_feature_names_out()

train_df = pd.DataFrame(
    X_train_processed,
    columns=feature_names
)
train_df[target] = y_train.to_numpy()

test_df = pd.DataFrame(
    X_test_processed,
    columns=feature_names
)
test_df[target] = y_test.to_numpy()

train_df.to_csv(
    RESULTS_DIR / "train_processed.csv",
    index=False
)
test_df.to_csv(
    RESULTS_DIR / "test_processed.csv",
    index=False
)

print("\nSaved processed datasets to results/")
print("Practical work completed successfully!")
