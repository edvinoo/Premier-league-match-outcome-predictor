import joblib
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from data import load_and_preprocess_data

def train_and_save_model():
    features_df, num_features, home_roll, away_roll = load_and_preprocess_data()

    X_teams = pd.get_dummies(features_df[['HomeTeam', 'AwayTeam']])
    X = pd.concat([X_teams, features_df[num_features]], axis=1)
    y = features_df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print("\nTuning hyperparameters...")
    base_model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        eval_metric='mlogloss',
        random_state=42
    )

    param_grid = {
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    print(f"\nTuned Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(f"Optimal Parameters: {grid_search.best_params_}")

    # Save artifacts to avoid re-running training
    joblib.dump(best_model, 'model.joblib')
    joblib.dump(list(X.columns), 'columns.joblib')
    joblib.dump(num_features, 'num_features.joblib')
    joblib.dump(home_roll, 'home_roll.joblib')
    joblib.dump(away_roll, 'away_roll.joblib')
    print("Model and metadata artifacts successfully saved.")

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Away Win', 'Draw', 'Home Win'])
    disp.plot(cmap='Blues')
    plt.title("XGBoost Confusion Matrix")
    plt.show()

if __name__ == '__main__':
    train_and_save_model()