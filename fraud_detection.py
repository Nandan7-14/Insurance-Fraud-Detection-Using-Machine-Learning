import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

df = pd.read_csv("dataset/insurance_claims.csv")

print(df.head())

print("Dataset Shape:", df.shape)


df.drop(['policy_number', '_c39'], axis=1, inplace=True)


df = df.replace('?', np.nan)


print(df.isnull().sum())


df = df.dropna()

print("New Dataset Shape:", df.shape)

le = LabelEncoder()

for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = le.fit_transform(df[column])

print(df.head())

sns.countplot(x='fraud_reported', data=df)

plt.title("Fraud vs Genuine Claims")

plt.show()

X = df.drop('fraud_reported', axis=1)

y = df['fraud_reported']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, pred_lr))

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
pred_dt = dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, pred_dt))

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, pred_rf))

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
pred_knn = knn.predict(X_test)
print("KNN Accuracy:", accuracy_score(y_test, pred_knn))

nb = GaussianNB()
nb.fit(X_train, y_train)
pred_nb = nb.predict(X_test)
print("Naive Bayes Accuracy:", accuracy_score(y_test, pred_nb))

svm = SVC()
svm.fit(X_train, y_train)
pred_svm = svm.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, pred_svm))

print("\nModel Comparison")

print("Logistic Regression:", accuracy_score(y_test, pred_lr))
print("Decision Tree:", accuracy_score(y_test, pred_dt))
print("Random Forest:", accuracy_score(y_test, pred_rf))
print("KNN:", accuracy_score(y_test, pred_knn))
print("Naive Bayes:", accuracy_score(y_test, pred_nb))
print("SVM:", accuracy_score(y_test, pred_svm))

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, pred_nb)

sns.heatmap(cm, annot=True, cmap="Blues")

plt.title("Confusion Matrix - Naive Bayes")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

import pickle

# Save trained model
with open("fraud_model.pkl", "wb") as f:
    pickle.dump(nb, f)







