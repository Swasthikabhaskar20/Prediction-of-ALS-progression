import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


# Specify the correct file path
file_path = 'C:/Users/swast/119.csv'

# Try reading with 'latin1' encoding
df = pd.read_csv(file_path, encoding='latin1')

progression_label = 'Diff'

# Feature Selection
X = df.drop(columns=[progression_label])
y = df[progression_label]

model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

feature_importance = pd.Series(model.feature_importances_, index=X.columns)
selected_features = feature_importance.nlargest(10).index.tolist()
print("Selected Features:", selected_features)

# Define base models
rf = RandomForestRegressor(n_estimators=100, max_depth=4, min_samples_split=10, 
                           min_samples_leaf=4, random_state=42, bootstrap=True) 
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, 
                               min_samples_split=5, min_samples_leaf=3, random_state=42)
svr = SVR(kernel='rbf', C=30, epsilon=0.05)

# Initialize MLPRegressor with chosen hyperparameters
mlp = MLPRegressor(hidden_layer_sizes=(100, 50), activation='tanh', solver='adam', 
                   learning_rate_init=0.0005, alpha=0.0001, batch_size=64, 
                   max_iter=1000, random_state=42, early_stopping=True)

# Train the MLP model
mlp.fit(X_train_scaled, y_train)

# Stacking Regressor with MLP as the final model
stacking = StackingRegressor(
    estimators=[('gb', gb), ('rf', rf), ('svr', svr)],
    final_estimator=mlp,
    cv=5
)

# Train the stacking model
stacking.fit(X_train_scaled, y_train)

# Predictions
y_train_pred_stacking = stacking.predict(X_train_scaled)
y_test_pred_stacking = stacking.predict(X_test_scaled)

# Model Evaluation
def evaluate_model(y_train, y_train_pred, y_test, y_test_pred, model_name):
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_train = np.sqrt(mse_train)
    rmse_test = np.sqrt(mse_test)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    
    print(f"{model_name} Performance:")
    print(f"  Training - MAE: {mae_train:.4f}, MSE: {mse_train:.4f}, RMSE: {rmse_train:.4f}, R²: {r2_train:.4f}")
    print(f"  Testing  - MAE: {mae_test:.4f}, MSE: {mse_test:.4f}, RMSE: {rmse_test:.4f}, R²: {r2_test:.4f}")






