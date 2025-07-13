This project builds and compares multiple machine learning models to predict house prices using the `Part1_house_price.csv` dataset.

### 📋 Steps
1. Data loading & exploration:
   - Missing values check
   - Correlation heatmap & price distribution plot
2. Feature engineering:
   - Extract `sale_year`, `sale_month` from `date`
   - Compute `house_age`, `years_since_renovation`, `price_per_sqft`
   - Drop irrelevant columns
3. Preprocessing:
   - Train-test split (70%-30%)
   - Standard scaling of numerical features
4. Model training & hyperparameter tuning:
   - Random Forest
   - Gradient Boosting
   - XGBoost
   - LightGBM
   - MLPRegressor
5. Evaluation:
   - Metrics: MSE, R², MAE
   - Feature importance & residual plots (Random Forest)

### 📦 Models & Libraries
- `scikit-learn`
- `xgboost`
- `lightgbm`
- `pandas`, `numpy`, `matplotlib`, `seaborn`

### 📈 Results
The script prints and compares performance metrics (MSE, R², MAE) for all models and shows feature importances & residuals for Random Forest.
