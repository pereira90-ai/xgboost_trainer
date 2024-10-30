import xgboost as xgb
from PySide6.QtWidgets import QMessageBox
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from xgboost import XGBClassifier
import pandas as pd


def read_dtaset(path = 'A_60_Historical_Data.csv'):
    if not path.endswith('.csv'):
        QMessageBox.warning("File Format Error", "The selected file is not a CSV.")
        return

    data = pd.read_csv(path)
    data = data[data['Title_ A_H'].str.len() <= 1]

    data['Win_Lose'] = data['Win_Lose'].apply(lambda x: 1 if 'win' in x.lower() else 0)
    data['WAX_wane'] = data['WAX_wane'].apply(lambda x: 1 if 'wax' in x.lower() else 0)
    data = data.reset_index(drop=True)
    X = data.drop(columns=['Day', 'Win_Lose', 'Value', 'Title_ A_H', 'Total'])
    y = data.Win_Lose
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_val, y_train, y_val


model = XGBClassifier(
    objective='binary:logistic',
    n_estimators=100,  # Here you specify the number of boosting rounds
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    early_stopping_rounds=10
)
#
# model.fit(
#     X_train,
#     y_train,
#     eval_set=[(X_val, y_val)],
#     verbose=True
# )


# proba_predictions = model.predict_proba(X)[:, 1]  # Get probabilities for the positive class