Insurance Fraud Detection Using Machine Learning
Project Overview

Insurance fraud is a major problem in the insurance industry. Fraudulent claims lead to significant financial losses and increase operational costs for insurance companies. Detecting fraudulent claims manually is difficult due to the large volume of insurance transactions and the complexity of fraud patterns.

This project applies machine learning techniques to analyze historical insurance claim data and automatically predict whether a claim is fraudulent or genuine. The goal is to assist insurance companies in identifying suspicious claims and improving claim verification processes.

Problem Statement

Insurance companies receive thousands of claims daily. Some claimants submit false or exaggerated claims to obtain financial benefits they are not entitled to. These fraudulent activities cause significant financial losses.

The objective of this project is to develop a machine learning-based fraud detection system that can analyze insurance claim data and classify claims as fraudulent or legitimate.

Dataset Description

The dataset used in this project contains historical insurance claim records.

Dataset Characteristics

Attribute	Description
Total Records	1000
Total Features	40
Target Variable	fraud_reported

Key Features in Dataset

Customer information (age, occupation, relationship)

Policy information (policy state, deductible, premium)

Incident details (incident type, severity, location)

Claim details (total claim amount, injury claim, property claim)

Vehicle details (make, model, year)

The target variable indicates whether a claim is fraudulent.

Value	Meaning
0	Genuine Claim
1	Fraudulent Claim
Project Workflow

The project follows a typical machine learning pipeline.

Dataset
   ↓
Data Cleaning
   ↓
Feature Encoding
   ↓
Exploratory Data Analysis
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Machine Learning Models
   ↓
Model Evaluation
   ↓
Fraud Prediction
Data Preprocessing

Before training machine learning models, the dataset was cleaned and prepared.

Steps Performed

Removed unnecessary columns such as policy_number and _c39.

Replaced missing values represented by "?" with NaN.

Removed rows containing missing values.

Converted categorical variables into numerical values using Label Encoding.

Applied StandardScaler for feature normalization.

These steps ensured that the dataset was suitable for machine learning algorithms.

Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand patterns and relationships in the dataset.

Key analyses included:

Distribution of fraudulent vs genuine claims

Claim amount distribution

Incident severity analysis

Age distribution of policyholders

Correlation analysis between features

EDA helps identify patterns that may indicate fraudulent behavior.

Machine Learning Models

Several classification algorithms were implemented to detect fraudulent insurance claims.

Logistic Regression

Logistic Regression is a statistical model used for binary classification. It predicts the probability that a claim belongs to a fraudulent class.

Decision Tree

Decision Tree is a rule-based model that splits data into branches based on feature values, forming a tree-like structure used for classification.

Random Forest

Random Forest is an ensemble learning algorithm that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

K-Nearest Neighbors (KNN)

KNN is a distance-based algorithm that classifies new data points based on the majority class of their nearest neighbors.

Naïve Bayes

Naïve Bayes is a probabilistic classifier based on Bayes' theorem. It assumes independence between features and calculates probabilities for classification.

Support Vector Machine (SVM)

SVM is a powerful classification algorithm that finds the optimal hyperplane separating fraudulent and genuine claims.

Model Performance

The performance of each model was evaluated using accuracy.

Model	Accuracy
Logistic Regression	0.75
Decision Tree	0.77
Random Forest	0.76
KNN	0.73
Naïve Bayes	0.79
SVM	0.79
Best Performing Models

Naïve Bayes and Support Vector Machine achieved the highest accuracy of 79%, making them the best-performing models for this dataset.

Literature Survey

Several studies have explored the use of machine learning techniques for detecting insurance fraud.

| Author                                                   | Year | Contribution                                                                                           |
| -------------------------------------------------------- | ---- | ------------------------------------------------------------------------------------------------------ |
| Viaene, S., Derrig, R., Baesens, B., & Dedene, G.        | 2004 | Applied Bayesian learning neural networks for automobile insurance fraud detection.                    |
| Phua, C., Lee, V., Smith, K., & Gayler, R.               | 2010 | Provided a comprehensive survey of fraud detection techniques using data mining and machine learning.  |
| Ngai, E. W. T., Hu, Y., Wong, Y. H., Chen, Y., & Sun, X. | 2011 | Reviewed data mining applications for fraud detection in financial and insurance sectors.              |
| Abdallah, A., Maarof, M., & Zainal, A.                   | 2016 | Presented a survey of fraud detection techniques using machine learning and anomaly detection methods. |
| Baesens, B., Van Vlasselaer, V., & Verbeke, W.           | 2015 | Demonstrated predictive analytics techniques for fraud detection in insurance and banking systems.     |


These studies show that machine learning algorithms such as Random Forest, SVM, and Bayesian models are effective in identifying fraudulent transactions.

Technologies Used
Technology	Purpose
Python	Programming language
Pandas	Data manipulation
NumPy	Numerical operations
Matplotlib	Data visualization
Seaborn	Statistical visualization
Scikit-learn	Machine learning algorithms


Project Structure
insurance-fraud-detection
│
├── dataset
│   └── insurance_claims.csv
│
├── fraud_detection.py
│
├── README.md
│
└── requirements.txt
Applications

The proposed system can be used by insurance companies to:

Detect fraudulent insurance claims

Reduce financial losses

Improve claim verification efficiency

Support data-driven decision making

Conclusion

This project implemented multiple machine learning algorithms to detect fraudulent insurance claims using historical insurance claim data.

Several classification models were trained and evaluated, including Logistic Regression, Decision Tree, Random Forest, KNN, Naïve Bayes, and SVM.

Among these models, Naïve Bayes and Support Vector Machine achieved the highest accuracy, demonstrating their effectiveness in detecting fraudulent insurance claims.

The developed system can help insurance companies identify suspicious claims early and improve fraud detection processes.