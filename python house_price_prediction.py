import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb

# Load dataset
data = pd.read_csv('Part1_house_price.csv')
print("First few rows:\n", data.head())
print("\nMissing values:\n", data.isnull().sum())

# EDA: Correlation & Price Distribution
numeric_data = data.select_dtypes(include=[np.number])
plt.figure(figsize=(10,8))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

sns.histplot(data['price'], kde=True, bins=50)
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Feature engineering
data['date'] = pd.to_datetime(data['date'], format='%Y%m%dT%H%M%S')
data['sale_year'] = data['date'].dt.year
data['sale_month'] = data['date'].dt.month
data['house_age'] = data['sale_year'] - data['yr_built']
data['years_since_renovation'] = data.apply(
    lambda row: row['sale_year'] - row['yr_renovated'] if row['yr_renovated'] != 0 else 0, axis=1)
data['price_per_sqft'] = data['price'] / data['sqft_living']

data.drop(columns=['id', 'date', 'yr_built', 'yr_renovated'], inplace=True)

# Train-test split
X = data.drop(columns=['price'])
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale numerical features
scaler = StandardScaler()
num_cols = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'lat', 'long', 
            'sqft_living15', 'sqft_lot15', 'house_age', 'years_since_renovation', 'price_per_sqft']
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Helper function for tuning & evaluation
def tune_and_evaluate(model, param_grid, X_train, y_train, X_test, y_test, name):
    grid = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    y_pred = best.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"\n{name} — MSE: {mse:.2f}, R²: {r2:.2f}, MAE: {mae:.2f}")
    print(f"Best params: {grid.best_params_}")
    return best, y_pred

# Models & parameter grids
models_params = [
    (RandomForestRegressor(random_state=42), {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None]
    }, "Random Forest"),
    (GradientBoostingRegressor(random_state=42), {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    }, "Gradient Boosting"),
    (xgb.XGBRegressor(random_state=42), {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    }, "XGBoost"),
    (lgb.LGBMRegressor(random_state=42), {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'num_leaves': [31, 50]
    }, "LightGBM"),
    (MLPRegressor(random_state=42, max_iter=500), {
        'hidden_layer_sizes': [(128,64), (256,128)],
        'activation': ['relu', 'tanh'],
        'learning_rate': ['constant', 'adaptive']
    }, "MLP Regressor")
]

# Run all models
predictions = {}
for model, params, name in models_params:
    best_model, y_pred = tune_and_evaluate(model, params, X_train, y_train, X_test, y_test, name)
    predictions[name] = (best_model, y_pred)

# Feature Importance (Random Forest)
rf_best = predictions['Random Forest'][0]
importances = rf_best.feature_importances_
indices = np.argsort(importances)
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel('Importance')
plt.title('Random Forest Feature Importance')
plt.show()

# Residual Plot (Random Forest)
residuals_rf = y_test - predictions['Random Forest'][1]
plt.scatter(y_test, residuals_rf)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Actual Price')
plt.ylabel('Residuals')
plt.title('Random Forest Residuals')
plt.show()
