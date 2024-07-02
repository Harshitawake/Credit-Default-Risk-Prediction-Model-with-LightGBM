# Credit Default Risk Prediction Model with LightGBM

This project aims to predict the probability of credit default using a LightGBM model. The goal is to build an effective machine learning model that can help financial institutions assess the risk associated with lending to customers.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Approach](#approach)
  - [Data Preprocessing](#data-preprocessing)
  - [Feature Engineering](#feature-engineering)
  - [Roll Rate Analysis](#roll-rate-analysis)
  - [Window Roll Rate Analysis](#window-roll-rate-analysis)
  - [Target Encoding](#target-encoding)
  - [Feature Selection](#feature-selection)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
  - [SHAP Analysis](#shap-analysis)
- [Why LightGBM?](#why-lightgbm)
- [Results](#results)
- [Usage](#usage)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)

## Introduction

Credit default risk prediction is crucial for financial institutions to manage and mitigate potential losses. This project leverages LightGBM, a gradient boosting framework, to build a predictive model that estimates the likelihood of a customer defaulting on their credit.

## Dataset

The dataset used for this project contains various features related to the customer's credit history, demographic information, and other financial attributes. The target variable indicates whether the customer has defaulted on their credit.

## Approach

### Data Preprocessing

Data preprocessing involves handling missing values, encoding categorical variables, and scaling numerical features. This step is essential to ensure the data is clean and suitable for model training.

### Feature Engineering

Feature engineering includes creating new features from existing ones to enhance the model's predictive power. This can involve combining multiple features, creating interaction terms, and extracting meaningful insights from raw data.

### Roll Rate Analysis

Roll rate analysis is used to understand the probability of a customer rolling from one delinquency state to another (e.g., from current to 30 days past due). This analysis helps identify patterns in customer payment behavior.

#### Example:

If a customer is 30 days past due (DPD), roll rate analysis can help determine the likelihood of this customer becoming 60 days past due.

### Window Roll Rate Analysis

Window roll rate analysis extends the roll rate analysis by considering a window of EMIs (Equated Monthly Installments). It helps in identifying the probability of a customer being `x` days past due within `y` EMIs.

#### Example:

For a customer with 3 EMIs:
- **1st EMI:** Current
- **2nd EMI:** 30 days past due (DPD)
- **3rd EMI:** 60 days past due (DPD)

Window roll rate analysis helps in understanding the likelihood of this customer being 60 days past due within 3 EMIs.

### Target Encoding

Target encoding is used to convert categorical variables into numerical values based on the target variable. This technique is particularly useful in handling high cardinality categorical features.

#### Benefits of Target Encoding:

1. **Handles High Cardinality:** Efficiently encodes categorical variables with a large number of unique values.
2. **Improves Predictive Power:** Utilizes the relationship between the categorical feature and the target variable to enhance model performance.
3. **Reduces Overfitting:** When combined with regularization techniques, target encoding can help prevent overfitting.

### Feature Selection

For feature selection, Random Forest and Decision Trees were used to determine feature importance. This method helps in identifying the most significant features that contribute to the prediction of credit default risk.

#### Benefits of Feature Importance Method:

1. **Identifies Key Features:** Helps in understanding which features are most impactful in the prediction model.
2. **Improves Model Performance:** By focusing on the most important features, the model can achieve better performance and generalization.
3. **Reduces Dimensionality:** Simplifies the model by eliminating less important features, reducing the risk of overfitting.

### Model Training

The model is trained using LightGBM, which is known for its efficiency and high performance on large datasets. The training process involves:

- Splitting the data into training and validation sets.
- Training the LightGBM model with optimal hyperparameters.
- Evaluating the model's performance on the validation set.

### Model Evaluation

Model evaluation metrics include:

- **Accuracy**: The proportion of correctly predicted instances out of the total instances.
- **Precision**: The proportion of true positive predictions out of all positive predictions.
- **Recall**: The proportion of true positive predictions out of all actual positives.
- **F1 Score**: The harmonic mean of precision and recall.
- **AUC-ROC**: The area under the receiver operating characteristic curve, which measures the model's ability to distinguish between classes.

### SHAP Analysis

SHAP (SHapley Additive exPlanations) was used to interpret the model's predictions. SHAP values provide a unified measure of feature importance and the effect of each feature on individual predictions.

#### Benefits of SHAP Analysis:

1. **Interpretable Predictions:** Helps in understanding the contribution of each feature to the final prediction.
2. **Model Transparency:** Provides insights into how the model makes decisions, enhancing trust in the model.
3. **Feature Impact:** Visualizes the impact of each feature on the prediction, allowing for better model diagnostics.

## Why LightGBM?

LightGBM was chosen for this project due to several reasons:

1. **Efficiency and Speed:** LightGBM is designed to be highly efficient, making it faster than many other gradient boosting algorithms, especially on large datasets.
2. **High Performance:** It often provides better performance and accuracy compared to other algorithms, thanks to its leaf-wise growth strategy.
3. **Handling Large Datasets:** LightGBM can handle large datasets with high dimensionality, making it suitable for complex tasks like credit default risk prediction.
4. **Support for Parallel and GPU Learning:** It supports parallel and GPU learning, which can significantly reduce training time.

## Results

The model achieves high performance in predicting credit default risk, with strong metrics across accuracy, precision, recall, F1 score, and AUC-ROC. The feature importance plot shows the most influential features in the prediction process.

## Usage

To use this model:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Harshitawake/Credit-Default-Risk-Prediction-Model-with-LightGBM.git
   cd Credit-Default-Risk-Prediction-Model
