# Prediction-of-ALS-progression

# 1.Enhancing ALS Progression Prediction using Stacking Multi-layer GSR Ensemble Model with XAI Visualization Techniques
This project implements a ALS progression prediction using stacking multi-layer GSR ensemble model using ALS datasets. It was developed as part of the research submitted to PeerJ.

# 2.Description - An overview of the code/dataset.
This project uses a regression model to predict the progression of ALS by applying a Stacking Regressor ensemble technique. The stacking model averages the outputs of  base regressors (e.g., Random Forest, Support Vector Regressor, and Gradient Boosting) using a meta-learner (MLP) to achieve higher accuracy and generalization. The data set contains ALS patient records with clinical features. This ensemble model is trained and validated, its performance is measured using metrics like Mean Squared Error (MSE) and R² score.

# 3.Dataset Information
Name: ALS Help Request to the Community<br> 
Format: Microsoft Excel Worksheet (.csv)<br> 
Source: Collected from  kaggle - https://www.kaggle.com/datasets/juanophillips/als-help-request-to-the-community  <br> 
Features: <br> 
SubjectUID - identification code for the predefines patient.<br> 
Age at Symptoms Onset -when the first symptoms started.<br> 
Death - if the patient already passed away.<br> 
Site of Onset - this disease can start in any part of the body, there it is detailed were in thar particular patient.<br> 
Cohort -  group of people carrying or related to the disease. ALS means confirmed diagnosis; Asymptomatic ALS Gene Carrier is someone who's got a gene mutation associated with ALS but at the screening baseline did not have a confirmed diagnosis; Non-ALS MND, similar diseases that carry some of the characteristics of ALS.<br> 
Med - Original med as reported in the original data base.<br> 
Med_Revised - standarized med names using Pubchem (National Center for Biotechnology Information) as a database to harmonize the names and components.<br> 
DiagDT -time elapsed between the confirmed diagnosis and the screening date. Negative number of 0,6767, means the patient was diagnosed 0,6767 years before the screening date (365 days base). Positive number means it happened after the screening date.<br> 
Onset Yr - time elapsed between the first symptoms and the screening date.<br> 
LNA_YR - last known alive date after the screening date.<br> 
Lenght_Diag_LNA - time elapsed between diagnose and Last Known Alive date.<br> 
ALSFRS-R Baseline - ALS functional rating scale, which is a standard metric to measure how people are, at the beginning of measurement, in terms of their muscular and other physical capabilities. (48 to 0), 48 is the best, cero is bad. <br> 
ALSFRS-R Latest - ALS functional rating scale, which is a standard metric to measure how people are, at the end of measurement, in terms of their motor capabilities. (48 to 0). <br> 
Diff - ALSFRS-R Latest minus ALSFRS-R Baseline. <br> 

# 4.Code Information
Language: Python <br>
Libraries:<br> 
pandas, numpy, matplotlib.pyplot, lime, lime.lime_tabular, sklearn.model_selection, sklearn.linear_model, sklearn.ensemble, sklearn.svm, sklearn.neural_network, sklearn.decomposition, sklearn.preprocessing, sklearn.metrics, xgboost. <br> <br> 
Goal: Predict ALS progression using a stacking ensemble of multiple regressors.<br>

Workflow:

Load and preprocess ALS dataset.<BR>
Define base regressors: Gradient Boosting Regressor, Support Vector Regressor,
and Random Forest Regressor.<BR>
Combine them using StackingRegressor with a MLP meta-learner.<BR>
Train the model and evaluate using R², MAE, and RMSE.<BR>
Output: Trained model file and printed evaluation metrics.<BR>


# 5.Usage Instructions – How to use or load the dataset and code
# Load the dataset
import pandas as pd

# Specify the correct file path
file_path = 'C:/Users/swast/ALS.csv'

# Try reading with 'latin1' encoding
data = pd.read_csv(file_path, encoding='latin1')

# 6. Requirements – Any dependencies (e.g., Python libraries).
# Import necessary libraries

import pandas as pd <br>
import numpy as np <br>
import matplotlib.pyplot as plt<br>
import lime<br>
import lime.lime_tabular<br>
from sklearn.model_selection import train_test_split<br>
from sklearn.linear_model import Lasso<br>
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, GradientBoostingRegressor<br>
from sklearn.svm import SVR<br>
from sklearn.neural_network import MLPRegressor<br>
from sklearn.decomposition import PCA<br>
from sklearn.preprocessing import StandardScaler<br>
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score<br>
from xgboost import XGBRegressor<br>
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Any dependencie
Programming Language:

Python 3.8 or later

Required Libraries:

pandas – for data manipulation and analysis

numpy – for numerical computing

matplotlib – for plotting and visualization

lime – for model interpretability using LIME (Local Interpretable Model-agnostic Explanations)

scikit-learn – for preprocessing, model training, evaluation, and decomposition

xgboost – for gradient boosting regression models

# 7. Methodology (if applicable) – Steps taken for data processing.
# Preprocessing
labelencoder is used for converting categorical columns to numerical values.
Non-numeric entries is changed to numberic values(eg:"-").

# Initialize a label encoder
le = LabelEncoder()

# Fit and transform the 'Site_of_Onset' column to numeric values
df['Site_of_Onset'] = le.fit_transform(df['Site_of_Onset'])


# Fit and transform the 'Site_of_Onset' column to numeric values
df['Cohort'] = le.fit_transform(df['Cohort'])


df['Lenght_Diag_LNA'] = pd.to_numeric(df['Lenght_Diag_LNA'], errors='coerce')

# Calculate the average of non-missing values
non_missing_avg = df['Lenght_Diag_LNA'].dropna().mean()

# Replace NaN (originally '-') with the rounded average
df['Lenght_Diag_LNA'] = df['Lenght_Diag_LNA'].fillna(round(non_missing_avg))

# Display the updated DataFrame
df
# Drop unneccessary columns
df.drop(columns=['SubjectUID'], inplace=True)
df.drop(columns=['Unnamed: 15'], inplace=True)

average_length = df['Lenght_Diag_LNA'].mean()

# Replace NaN with the rounded average
df['Lenght_Diag_LNA'] = df['Lenght_Diag_LNA'].fillna(round(average_length))

# Display the updated DataFrame
df

df['Diagdt'] = pd.to_numeric(df['Diagdt'], errors='coerce')

# Calculate the average of non-missing values
non_missing_avg = df['Diagdt'].dropna().mean()

# Replace NaN (originally '-') with the rounded average
df['Diagdt'] = df['Diagdt'].fillna(round(non_missing_avg))


df['Age at Symptons Onset'] = pd.to_numeric(df['Age at Symptons Onset'], errors='coerce')

# Calculate the average of non-missing values
non_missing_avg = df['Age at Symptons Onset'].dropna().mean()

# Replace NaN (originally '-') with the rounded average
df['Age at Symptons Onset'] = df['Age at Symptons Onset'].fillna(round(non_missing_avg))

# Display the updated DataFrame
df['Site_of_Onset'] = pd.to_numeric(df['Site_of_Onset'], errors='coerce')

# Calculate the average of non-missing values
non_missing_avg = df['Site_of_Onset'].dropna().mean()

# Replace NaN (originally '-') with the rounded average
df['Site_of_Onset'] = df['Site_of_Onset'].fillna(round(non_missing_avg))

# Display the updated DataFrame



plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.show()


Fills Nan values with average values to clean the dataset.

Detects Outliers using IQR rule for medical dataset(ALS).

Removes Outliers outside the boundaries and visuvalized using Scatter plot.

# Function to detect outliers using IQR
def detect_outliers_iqr(data, column):<BR>
    Q1 = data[column].quantile(0.25)<BR>
    Q3 = data[column].quantile(0.75)<BR>
    IQR = Q3 - Q1<BR>
    lower_bound = Q1 - 1.5 * IQR<BR>
    upper_bound = Q3 + 1.5 * IQR<BR>
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]<BR>
    return outliers, lower_bound, upper_bound<BR>

# Loop through all selected features
for col in columns_to_check:<BR>
    outliers, lower_bound, upper_bound = detect_outliers_iqr(df, col)<BR>
    print(f"Feature: {col} → {len(outliers)} Outliers Detected")<BR>
    plt.figure(figsize=(8, 5))<BR>
    sns.boxplot(x=df[col], color='lightblue')<BR>
    plt.axvline(lower_bound, color='red', linestyle='dashed', label='Lower Bound')<BR>
    plt.axvline(upper_bound, color='red', linestyle='dashed', label='Upper Bound')<BR>
    plt.scatter(outliers[col], np.ones(len(outliers)), color='red', label='Outliers', zorder=3)<BR>
    plt.legend()<BR>
    plt.title(f"Boxplot with IQR Outliers for {col}")<BR>
    plt.show()<BR>

Missing values are handled.

# Feature Selection using XGBoost model 

Base-model
Random Forest Regressor(RF) is used for handling non-linear data,Gradient Boosting Regressor (GB) improves predictions and Support Vector Regressor (SVR) for scaled numerical features.
MLP- powerful deep learning model.


X = df.drop(columns=[progression_label])<br>
y = df[progression_label]<br>

model = XGBRegressor(n_estimators=100, random_state=42)<br>
model.fit(X, y)<br>

feature_importance = pd.Series(model.feature_importances_, index=X.columns)<br>
selected_features = feature_importance.nlargest(10).index.tolist()<br>
print("Selected Features:", selected_features)<br>
# 7. Methodology (if applicable) – Steps taken for modeling. 
# Train Model
# Define base models
rf = RandomForestRegressor(n_estimators=100, max_depth=4, min_samples_split=10, 
                           min_samples_leaf=4, random_state=42, bootstrap=True) <br>
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, 
                               min_samples_split=5, min_samples_leaf=3, random_state=42)<br>
svr = SVR(kernel='rbf', C=30, epsilon=0.05)<br>

# Initialize MLPRegressor with specific hyperparameters
mlp = MLPRegressor(hidden_layer_sizes=(100, 50), activation='tanh', solver='adam', 
                   learning_rate_init=0.0005, alpha=0.0001, batch_size=64, 
                   max_iter=1000, random_state=42, early_stopping=True)<br>

# Train the MLP model
mlp.fit(X_train_scaled, y_train)<br>

# Stacking Regressor with MLP as the final model
stacking = StackingRegressor(
    estimators=[('gb', gb), ('rf', rf), ('svr', svr)],
    final_estimator=mlp,
    cv=5
)<br>

# Train the stacking model
stacking.fit(X_train_scaled, y_train)<br>

# Predictions
y_train_pred_stacking = stacking.predict(X_train_scaled)<br>
y_test_pred_stacking = stacking.predict(X_test_scaled)<br>

# Model Evaluation
Evaluation metrics used here is
- *MSE (Mean Squared Error)*
- *RMSE (Root Mean Squared Error)*
- *MAE (Mean Absolute Error)*
- *R² Score (Coefficient of Determination)*

print(f"{model_name} Performance:")
print(f"  Training - MAE: {mae_train:.4f}, MSE: {mse_train:.4f}, RMSE: {rmse_train:.4f}, R²: {r2_train:.4f}")
print(f"  Testing  - MAE: {mae_test:.4f}, MSE: {mse_test:.4f}, RMSE: {rmse_test:.4f}, R²: {r2_test:.4f}")

# XAI- Visuvalizing 
LIME (Local Interpretable Model-agnostic Explanations) to explain the predictions of a regression model.

Select any instance 

# Create LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train_scaled,  # Use training data for sampling
    feature_names=selected_features,  # Feature names
    class_names=['Prediction'],  # Single output regression
    mode='regression'
)

# Explain the prediction for the chosen test instance
explanation = explainer.explain_instance(
    X_test_scaled[sample_index],  # The instance to explain
    stacking.predict,  # The trained model's predict function
    num_features=10  # Number of top influential features
)

# 9.License & Contribution Guidelines
# ALS Progression Prediction Code for PeerJ Submission

This repository contains source code related to the research article submitted to PeerJ.

## License & Restrictions

The code is provided for the sole purpose of **peer review**.  
**Reuse, redistribution, or modification of this code is NOT permitted** without the explicit written permission of the author.

© [Swasthika Bhaskar], [2025]
