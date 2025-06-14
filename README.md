# Salary Estimator

## Project Summary

**Salary Estimator** is a machine learning project that predicts whether an individual's income exceeds $50K/year based on demographic and employment attributes. The project uses the UCI "Adult" dataset and implements a comprehensive suite of machine learning models, including linear models, tree-based models, distance-based models, support vector machines, Bayesian models, and neural networks. The notebook provides data preprocessing, model training, evaluation, and visualization of results.

---

## Project Details

### Dataset

- **Source:** [UCI Machine Learning Repository: Adult Data Set](https://doi.org/10.24432/C5XW20)
- **Description:** The dataset contains census data with features such as age, workclass, education, occupation, race, sex, capital gain/loss, hours per week, native country, and income label.

### Problem Statement

Given a set of features for an individual, predict whether their income is `<=50K` or `>50K`.

---

## Project Structure
---

salary_estimator/ 
│ 
├── salary_estimator.ipynb # Main Jupyter notebook with code, analysis, and results 
├── README.md # Project documentation 
├── LICENSE # MIT License 
├── adult.data # Training data (not included in repo) 
└── adult.test # Test data (not included in repo)

---

## Tools & Libraries Used

- **Python 3**
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical operations
- **Matplotlib**: Data visualization
- **scikit-learn**: Machine learning models and preprocessing
  - Linear Models: `LogisticRegression`, `SGDClassifier`, `RidgeClassifier`
  - Tree-Based Models: `DecisionTreeClassifier`, `RandomForestClassifier`
  - Distance-Based Models: `KNeighborsClassifier`, `RadiusNeighborsClassifier`
  - SVMs: `LinearSVC`, `SVC`
  - Bayesian Models: `GaussianNB`, `BernoulliNB`
  - Neural Networks: `MLPClassifier`
  - Preprocessing: `StandardScaler`
  - Metrics: `accuracy_score`, `precision_score`, `recall_score`, `f1_score`
- **XGBoost**: `XGBClassifier`
- **LightGBM**: `LGBMClassifier`
- **TensorFlow / Keras**: Deep neural networks

---

## Workflow Overview

1. **Data Loading**
   - Read training and test data from CSV files.
   - Assign column names.

2. **Data Exploration**
   - Display unique values for each feature.
   - Understand data distribution and missing values.

3. **Data Preprocessing**
   - Map categorical variables to numerical values using custom dictionaries.
   - Fill missing values with zeros.
   - Standardize features using `StandardScaler`.

4. **Model Training & Evaluation**
   - Train multiple models:
     - Linear: Logistic Regression, SGD, Ridge
     - Tree-based: Decision Tree, Random Forest, XGBoost, LightGBM
     - Distance-based: KNN, Radius Neighbors
     - SVM: Linear and Kernel
     - Bayesian: GaussianNB, BernoulliNB
     - Neural Networks: MLP, Deep Neural Network (Keras)
   - Evaluate each model using accuracy, precision, recall, and F1 score.
   - Print and visualize metrics for comparison.

5. **Visualization**
   - Plot individual and comparative bar charts for model metrics.

---

## How to Run

1. **Clone the repository** and ensure you have the required data files (`adult.data`, `adult.test`) in the project directory.

2. **Install dependencies**:
   ```sh
   pip install pandas numpy matplotlib scikit-learn xgboost lightgbm tensorflow

3. **Open salary_estimator.ipynb** in Jupyter Notebook or VS Code.

4. **Run all** cells to execute the workflow from data loading to model evaluation and visualization.

---

## Results

- The notebook provides a detailed comparison of various machine learning models on the salary prediction task.

- Performance metrics (accuracy, precision, recall, F1) are reported and visualized for each model.

- The project demonstrates the impact of different algorithms and preprocessing steps on classification performance.
