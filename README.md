# House Price Prediction System - California Housing Dataset

## AI & ML Assignment - Task 2: Feature Engineering, Model Optimization & Performance Comparison

This project implements a complete, professional, and well-documented House Price Prediction System using the **California Housing Dataset** and **scikit-learn**.

It evaluates and compares multiple regression models—Linear Regression, Ridge Regression, and Decision Tree Regressor—and optimizes their performance using cross-validated hyperparameter grid searches.

---

## Project Structure

```text
task2_ml/
│
├── task2_ml.ipynb          # The main Jupyter Notebook (pre-executed with all charts & metrics)
├── requirements.txt        # Required python packages for running the project
├── README.md               # Project documentation & summary of results (this file)
│
# Developer generation scripts (can be safely ignored or used for regeneration):
├── generate_and_run.py     # Python script to programmatically build and run the notebook
└── generate_notebook.py    # Python script to generate the notebook JSON structure
```

---

## Installation & Setup

To run the notebook locally, follow these steps:

1. **Clone or Navigate** to this directory:
   ```bash
   cd task2_ml
   ```

2. **Create and Activate a Virtual Environment** (Optional but recommended):
   ```bash
   python -m venv venv
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook task2_ml.ipynb
   # OR launch JupyterLab
   jupyter lab
   ```

---

## Methodology Overview

1. **Step 1: Import Libraries**: Import data science modules (`numpy`, `pandas`, `matplotlib`, `seaborn`) and scikit-learn metrics, preprocessings, and regressors.
2. **Step 2: Load the Dataset**: Import the California Housing dataset, translate features, output shape, info, first 5 rows, and descriptive summaries.
3. **Step 3: Exploratory Data Analysis (EDA)**: Detect null values, duplicates, display statistical properties, feature histograms, and a correlation matrix.
4. **Step 4: Feature Preprocessing & Scaling**: Extract input features and target value. Normalize features to zero-mean and unit-variance using `StandardScaler` to ensure numerical stability and correct regularization penalties.
5. **Step 5: Dataset Splitting**: Divide data into 80% train and 20% test subsets using a seed of `42`.
6. **Step 6: Train Baseline Models**: Fit Linear Regression, Ridge Regression ($\alpha=1.0$), and a default Decision Tree Regressor.
7. **Step 7: Hyperparameter Tuning**: Perform a 5-fold cross-validated grid search (`GridSearchCV`) to optimize Ridge and Decision Tree hyperparameters.
8. **Step 8: Performance Evaluation**: Evaluate models on both train and test partitions using $R^2$, MSE, and MAE.
9. **Step 9: Performance Visualization**: Display bar chart comparisons of Test metrics and plot Actual vs. Predicted values.
10. **Step 10: Conclusion & Insights**: Analyze the performance difference between linear and tree-based methods and the importance of tree pruning.

---

## Evaluation Results Summary

The evaluation results of all baseline and optimized models are summarized below:

| Model | Train $R^2$ | Test $R^2$ | Train MSE | Test MSE | Train MAE | Test MAE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | 0.613 | 0.576 | 0.518 | 0.556 | 0.529 | 0.533 |
| **Ridge Regression ($\alpha=1.0$)** | 0.613 | 0.576 | 0.518 | 0.556 | 0.529 | 0.533 |
| **Tuned Ridge Regression** | 0.613 | 0.576 | 0.518 | 0.556 | 0.529 | 0.533 |
| **Baseline Decision Tree** | 1.000 | 0.624 | 0.000 | 0.493 | 0.000 | 0.453 |
| **Tuned Decision Tree** | **0.818** | **0.693** | **0.243** | **0.403** | **0.340** | **0.430** |

### Key Findings:
- **Decision Trees capture non-linearities**: Standard linear models achieve a Test $R^2$ of **0.576**. The Tuned Decision Tree achieves a Test $R^2$ of **0.693**, indicating that housing price dynamics are heavily non-linear (e.g. geographical coordinates).
- **Avoiding Overfitting**: The default Decision Tree overfits completely, achieving a Train $R^2$ of **1.000** but a Test $R^2$ of only **0.624**. Grid Search optimization limited the tree depth, which significantly increased the Test $R^2$ score to **0.693** and lowered the Test MSE to **0.403**.
