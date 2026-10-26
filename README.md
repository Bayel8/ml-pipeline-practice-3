
# Practical Work 3: Preprocessing Pipeline

## Objective
Build a preprocessing pipeline and prevent data leakage between training and test data.

## Dataset
Titanic passenger dataset.

Target variable: `Survived`.

## Features
- Numerical: Age, Fare, SibSp, Parch
- Categorical: Pclass, Sex, Embarked

## Preprocessing strategies
- Numerical missing values: median imputation
- Pclass missing values: most frequent category
- Sex and Embarked missing values: separate category Unknown
- Numerical features: StandardScaler
- Categorical features: OneHotEncoder

## Leakage prevention
The dataset is split before preprocessing.
The preprocessor is fitted only on training data.
The test data is transformed using the fitted preprocessor.

## Installation
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run
```powershell
python main.py
```

## Results
Processed training and test datasets are saved in the results folder.
