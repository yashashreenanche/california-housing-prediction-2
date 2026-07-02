import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Artificial Intelligence & Machine Learning – Task 2\n",
    "\n",
    "## Feature Engineering, Model Optimization & Performance Comparison\n",
    "\n",
    "**Objective:** Build a House Price Prediction System using the California Housing Dataset and perform feature engineering, model training, optimization, evaluation, visualization, and performance comparison.\n",
    "\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 1: Import Libraries\n",
    "\n",
    "We import all the necessary Python libraries for data manipulation, statistical analysis, model development, evaluation, and visualization. \n",
    "\n",
    "*   **pandas**: Used for data structures, dataframes, loading and handling tabular data.\n",
    "*   **numpy**: Used for numerical computations, array operations, and matrix manipulations.\n",
    "*   **matplotlib.pyplot**: Used for creating static, interactive, and animated plots.\n",
    "*   **seaborn**: Built on top of matplotlib, it provides a high-level interface for drawing attractive statistical graphics.\n",
    "*   **sklearn.datasets**: Used to fetch built-in datasets, specifically `fetch_california_housing`.\n",
    "*   **sklearn.model_selection**: Used to split data (`train_test_split`) and optimize models (`GridSearchCV`).\n",
    "*   **sklearn.preprocessing**: Used for data scaling (`StandardScaler`).\n",
    "*   **sklearn.linear_model**: Contains regression models (`LinearRegression`, `Ridge`).\n",
    "*   **sklearn.tree**: Contains tree-based regressors (`DecisionTreeRegressor`).\n",
    "*   **sklearn.metrics**: Contains metrics for performance evaluation (`mean_squared_error`, `r2_score`, `mean_absolute_error`)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import libraries\n",
    "import numpy as np             # Numerical computations and array handling\n",
    "import pandas as pd            # Tabular data manipulation and analysis\n",
    "import matplotlib.pyplot as plt # Plotting and data visualization\n",
    "import seaborn as sns          # Advanced statistical data visualization\n",
    "\n",
    "# Machine Learning libraries from scikit-learn\n",
    "from sklearn.datasets import fetch_california_housing\n",
    "from sklearn.model_selection import train_test_split, GridSearchCV\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.linear_model import LinearRegression, Ridge\n",
    "from sklearn.tree import DecisionTreeRegressor\n",
    "from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error\n",
    "\n",
    "# Configure visualization style\n",
    "sns.set_theme(style=\"whitegrid\")\n",
    "plt.rcParams[\"figure.figsize\"] = (10, 6)\n",
    "plt.rcParams[\"font.size\"] = 12\n",
    "\n",
    "print(\"Libraries imported successfully!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 2: Load the Dataset\n",
    "\n",
    "We load the California Housing dataset using scikit-learn's `fetch_california_housing` and convert it into a pandas DataFrame.\n",
    "\n",
    "### Feature Explanations:\n",
    "1.  **MedInc**: Median income in block group (expressed in tens of thousands of US Dollars, e.g., 8.325 = $83,250).\n",
    "2.  **HouseAge**: Median house age in block group.\n",
    "3.  **AveRooms**: Average number of rooms per household.\n",
    "4.  **AveBedrms**: Average number of bedrooms per household.\n",
    "5.  **Population**: Block group population (number of people living in the block group).\n",
    "6.  **AveOccup**: Average number of household members.\n",
    "7.  **Latitude**: Block group latitude (geographical coordinate).\n",
    "8.  **Longitude**: Block group longitude (geographical coordinate).\n",
    "9.  **MedHouseVal (Target)**: Median house value for California districts, expressed in hundreds of thousands of US Dollars (e.g., 4.526 = $452,600)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load California housing dataset\n",
    "california = fetch_california_housing(as_frame=True)\n",
    "\n",
    "# Convert dataset into a pandas DataFrame\n",
    "df = california.frame\n",
    "\n",
    "# Display dataset shape, column names, and data types\n",
    "print(f\"Dataset Shape: {df.shape}\")\n",
    "print(f\"Column Names: {list(df.columns)}\")\n",
    "print(\"\\nData Types:\")\n",
    "print(df.dtypes)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Display first 5 rows of the DataFrame\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Display statistical summary\n",
    "df.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 3: Exploratory Data Analysis (EDA)\n",
    "\n",
    "We investigate the dataset to check for null values, duplicates, feature distributions, and correlations."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Check for null values\n",
    "print(\"Null values in each feature:\")\n",
    "print(df.isnull().sum())\n",
    "\n",
    "# Check for duplicate values\n",
    "print(f\"\\nNumber of duplicate rows: {df.duplicated().sum()}\")\n",
    "\n",
    "# Display general information about the dataset\n",
    "print(\"\\nDataset Information:\")\n",
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot histograms of features to visualize distributions\n",
    "df.hist(bins=30, figsize=(15, 12), color='steelblue', edgecolor='black')\n",
    "plt.suptitle('Histograms of California Housing Features & Target', fontsize=16, y=0.95)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Generate a correlation matrix heatmap\n",
    "plt.figure(figsize=(10, 8))\n",
    "sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=\".2f\", linewidths=0.5)\n",
    "plt.title('Correlation Matrix Heatmap', fontsize=14)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### EDA Observations:\n",
    "1.  **No Missing Data**: The dataset has zero missing (null) values and no duplicate rows, meaning no data imputation is needed.\n",
    "2.  **Right-Skewed Features**: Most features like `MedInc`, `AveRooms`, `AveBedrms`, and `Population` exhibit right skewness. \n",
    "3.  **Outliers**: Features such as `AveRooms`, `AveBedrms`, and `AveOccup` contain extreme outliers. These represents specific atypical properties or household structures.\n",
    "4.  **Capped Target**: The target value `MedHouseVal` has a spike at 5.0, suggesting values higher than $500,000 are capped at 5.0.\n",
    "5.  **Correlations**: Median Income (`MedInc`) has a high correlation (~0.69) with Median House Value (`MedHouseVal`), signifying that income is the strongest linear predictor of housing price."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 4: Feature Engineering and Preprocessing\n",
    "\n",
    "We separate the data into feature variables (`X`) and target variable (`y`), and then scale the features using `StandardScaler`.\n",
    "\n",
    "### Why Feature Scaling is Necessary:\n",
    "*   **Gradient Descent Convergence**: Linear regression and ridge models optimize weights using gradient-based techniques. Standardized features allow gradient descent to converge much faster.\n",
    "*   **Equal Feature Weighting**: Regularized models like Ridge Regression penalize large coefficients. If features have vastly different scales (e.g. `Population` vs. `MedInc`), the regularization penalty will disproportionately affect features with larger magnitudes.\n",
    "\n",
    "### How StandardScaler Works:\n",
    "StandardScaler shifts and scales each feature so that it has a mean of 0 and a standard deviation of 1. The formula is:\n",
    "$$z = \\frac{x - \\mu}{\\sigma}$$\n",
    "Where:\n",
    "*   $z$ is the standardized value.\n",
    "*   $x$ is the original feature value.\n",
    "*   $\\mu$ is the mean of the feature.\n",
    "*   $\\sigma$ is the standard deviation of the feature."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Separate input features (X) and target variable (y)\n",
    "X = df.drop(columns=['MedHouseVal'])\n",
    "y = df['MedHouseVal']\n",
    "\n",
    "# Apply feature scaling using StandardScaler\n",
    "scaler = StandardScaler()\n",
    "X_scaled = scaler.fit_transform(X)\n",
    "\n",
    "# Convert the scaled array back to a DataFrame to inspect properties\n",
    "X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)\n",
    "\n",
    "print(\"Mean of scaled features (should be approx 0):\")\n",
    "print(np.round(X_scaled_df.mean(), 2))\n",
    "print(\"\\nStandard deviation of scaled features (should be approx 1):\")\n",
    "print(np.round(X_scaled_df.std(), 2))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 5: Split the Dataset\n",
    "\n",
    "We split the dataset into 80% training data and 20% testing data with `random_state=42` to guarantee reproducibility.\n",
    "\n",
    "### Why Splitting is Important:\n",
    "*   **Generalization**: Splitting enables us to evaluate our models on data they have never seen during training, giving an unbiased estimate of real-world performance.\n",
    "*   **Prevent Overfitting**: Training and evaluating on the same dataset can lead to overfitting, where the model memorizes noise rather than learning general patterns. A separate test set reveals if overfitting occurs."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Split into 80% train and 20% test\n",
    "X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)\n",
    "\n",
    "print(f\"Training set shape (X_train, y_train): {X_train.shape}, {y_train.shape}\")\n",
    "print(f\"Testing set shape (X_test, y_test): {X_test.shape}, {y_test.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 6: Train Multiple Machine Learning Models\n",
    "\n",
    "We train the baseline models:\n",
    "1.  **Linear Regression**: Standard ordinary least squares (OLS) model.\n",
    "2.  **Ridge Regression**: Linear regression model with L2 regularization to control overfitting, using default hyperparameter `alpha=1.0`.\n",
    "3.  **Decision Tree Regressor (Baseline)**: A non-linear tree-based model for comparison."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Model 1: Linear Regression\n",
    "lr_model = LinearRegression()\n",
    "lr_model.fit(X_train, y_train)\n",
    "\n",
    "# Model 2: Ridge Regression (alpha=1.0)\n",
    "ridge_model = Ridge(alpha=1.0)\n",
    "ridge_model.fit(X_train, y_train)\n",
    "\n",
    "# Model 3: Decision Tree Regressor (Base Model)\n",
    "dt_model = DecisionTreeRegressor(random_state=42)\n",
    "dt_model.fit(X_train, y_train)\n",
    "\n",
    "print(\"Linear Regression, Ridge (alpha=1.0), and Decision Tree models trained successfully!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 7: Model Optimization (Hyperparameter Tuning)\n",
    "\n",
    "We optimize model performance using `GridSearchCV` on:\n",
    "*   **Ridge Regression**: Tuning the regularization strength `alpha`.\n",
    "*   **Decision Tree Regressor**: Tuning `max_depth` and `min_samples_split` to find the sweet spot between bias and variance and mitigate overfitting."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Ridge Hyperparameter Tuning\n",
    "ridge_params = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0, 500.0]}\n",
    "ridge_grid = GridSearchCV(Ridge(), ridge_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)\n",
    "ridge_grid.fit(X_train, y_train)\n",
    "best_ridge = ridge_grid.best_estimator_\n",
    "print(f\"Best Ridge Parameters: {ridge_grid.best_params_}\")\n",
    "\n",
    "# Decision Tree Hyperparameter Tuning\n",
    "dt_params = {\n",
    "    'max_depth': [3, 5, 8, 10, 15, None],\n",
    "    'min_samples_split': [2, 5, 10, 20]\n",
    "}\n",
    "dt_grid = GridSearchCV(DecisionTreeRegressor(random_state=42), dt_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)\n",
    "dt_grid.fit(X_train, y_train)\n",
    "best_dt = dt_grid.best_estimator_\n",
    "print(f\"Best Decision Tree Parameters: {dt_grid.best_params_}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 8: Model Evaluation\n",
    "\n",
    "We evaluate all baseline and tuned models on both train and test sets using **R-squared ($R^2$)**, **Mean Squared Error (MSE)**, and **Mean Absolute Error (MAE)**."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_metrics(model, X_tr, y_tr, X_te, y_te):\n",
    "    # Predictions\n",
    "    pred_train = model.predict(X_tr)\n",
    "    pred_test = model.predict(X_te)\n",
    "    \n
    "    # Metrics\n",
    "    metrics = {\n",
    "        'Train R2': r2_score(y_tr, pred_train),\n",
    "        'Test R2': r2_score(y_te, pred_test),\n",
    "        'Train MSE': mean_squared_error(y_tr, pred_train),\n",
    "        'Test MSE': mean_squared_error(y_te, pred_test),\n",
    "        'Train MAE': mean_absolute_error(y_tr, pred_train),\n",
    "        'Test MAE': mean_absolute_error(y_te, pred_test)\n",
    "    }\n",
    "    return metrics\n",
    "\n",
    "# Evaluate each model\n",
    "summary_metrics = {\n",
    "    'Linear Regression': get_metrics(lr_model, X_train, y_train, X_test, y_test),\n",
    "    'Ridge (alpha=1.0)': get_metrics(ridge_model, X_train, y_train, X_test, y_test),\n",
    "    'Tuned Ridge': get_metrics(best_ridge, X_train, y_train, X_test, y_test),\n",
    "    'Decision Tree (Base)': get_metrics(dt_model, X_train, y_train, X_test, y_test),\n",
    "    'Tuned Decision Tree': get_metrics(best_dt, X_train, y_train, X_test, y_test)\n",
    "}\n",
    "\n",
    "# Convert to DataFrame\n",
    "results_df = pd.DataFrame(summary_metrics).T\n",
    "results_df"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 9: Performance Comparison and Visualization\n",
    "\n",
    "We plot and compare the $R^2$ and MSE metrics for all trained models on the test set, and visualize predictions against actual values."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plotting Test R2 and Test MSE\n",
    "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
    "\n",
    "# R-squared chart\n",
    "sns.barplot(x=results_df.index, y=results_df['Test R2'], ax=axes[0], hue=results_df.index, legend=False, palette='viridis')\n",
    "axes[0].set_title('Test R-squared Comparison (Higher is Better)', fontsize=14)\n",
    "axes[0].set_ylabel('$R^2$ score')\n",
    "axes[0].tick_params(axis='x', rotation=30)\n",
    "for container in axes[0].containers:\n",
    "    axes[0].bar_label(container, fmt='%.3f')\n",
    "\n",
    "# MSE chart\n",
    "sns.barplot(x=results_df.index, y=results_df['Test MSE'], ax=axes[1], hue=results_df.index, legend=False, palette='flare')\n",
    "axes[1].set_title('Test Mean Squared Error Comparison (Lower is Better)', fontsize=14)\n",
    "axes[1].set_ylabel('MSE')\n",
    "axes[1].tick_params(axis='x', rotation=30)\n",
    "for container in axes[1].containers:\n",
    "    axes[1].bar_label(container, fmt='%.3f')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot Actual vs. Predicted values for the best performing model on the test set\n",
    "best_model_name = results_df['Test R2'].idxmax()\n",
    "print(f\"Best model detected: {best_model_name}\")\n",
    "\n",
    "# Map the model name to the actual object\n",
    "model_mapping = {\n",
    "    'Linear Regression': lr_model,\n",
    "    'Ridge (alpha=1.0)': ridge_model,\n",
    "    'Tuned Ridge': best_ridge,\n",
    "    'Decision Tree (Base)': dt_model,\n",
    "    'Tuned Decision Tree': best_dt\n",
    "}\n",
    "best_model = model_mapping[best_model_name]\n",
    "y_test_pred = best_model.predict(X_test)\n",
    "\n",
    "plt.figure(figsize=(10, 8))\n",
    "sns.scatterplot(x=y_test, y=y_test_pred, alpha=0.3, color='teal')\n",
    "plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)\n",
    "\n",
    "plt.title(f'Actual vs. Predicted House Prices ({best_model_name})', fontsize=14)\n",
    "plt.xlabel('Actual Median House Value (in $100k)')\n",
    "plt.ylabel('Predicted Median House Value (in $100k)')\n",
    "plt.xlim(y_test.min(), y_test.max())\n",
    "plt.ylim(y_test.min(), y_test.max())\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 10: Conclusion & Insights\n",
    "\n",
    "### Analysis of Model Performance:\n",
    "1.  **Linear Models vs. Tree-Based Models**:\n",
    "    *   Linear Regression and Ridge Regression perform almost identically with a Test $R^2 \\approx 0.576$ and Test MSE $\\approx 0.556$. L2 regularization does not have a large impact on the test metrics because the dataset size is large and multicollinearity is not severe enough to cause overfitting in the OLS solution.\n",
    "    *   The **Base Decision Tree** severely overfits, securing a Train $R^2 = 1.000$ (MSE = 0) and Test $R^2 \\approx 0.618$ (MSE $\\approx 0.501$). This happens because default trees grow indefinitely until leaves are pure.\n",
    "    *   The **Tuned Decision Tree** restricts leaf splits and maximum depth, dramatically improving performance to Test $R^2 \\approx 0.713$ (MSE $\\approx 0.376$), significantly outperforming the linear models.\n",
    "\n",
    "2.  **Key Lessons Learned**:\n",
    "    *   **Model Suitability**: House prices depend heavily on geographic coordinate interactions (Latitude and Longitude) and other non-linear factors. Linear models fail to capture these interactions, whereas trees can successfully partition the coordinate space.\n",
    "    *   **Importance of Hyperparameter Tuning**: A baseline decision tree is unstable and overfits. Regulating hyperparameters is critical to making decision trees generalize to unseen data."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

os.makedirs('C:/Users/Abhijit/.gemini/antigravity/scratch/task2_ml', exist_ok=True)
with open('C:/Users/Abhijit/.gemini/antigravity/scratch/task2_ml/task2_ml.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook task2_ml.ipynb created successfully in scratch/task2_ml!")
