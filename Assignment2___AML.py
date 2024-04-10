### Decision Tree ###

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

titanic = pd.read_csv("Titanic-Dataset.csv")
print(titanic.head())

titanic['Age'].fillna(titanic['Age'].median(), inplace=True)
titanic['Sex'] = titanic['Sex'].map({'male': 0, 'female': 1})

X = titanic[['Sex', 'Age']]
y = titanic['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)
# Make predictions on the test set
y_pred = decision_tree.predict(X_test)

#accuracy = accuracy_score(y_test, y_pred)
#print("Accuracy:", accuracy)

print(classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


#***************************************************************************************************************************#
### SVM ###

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

titanic = pd.read_csv("Titanic-Dataset.csv")
titanic['Age'].fillna(titanic['Age'].median(), inplace=True)
titanic['Sex'] = titanic['Sex'].map({'male': 0, 'female': 1})

X = titanic[['Sex', 'Age']]
y = titanic['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

#accuracy = accuracy_score(y_test, y_pred)
#print("Accuracy:", accuracy)

print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

#***************************************************************************************************************************#
### Logistic Regression ###

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

titanic = pd.read_csv("Titanic-Dataset.csv")
print(titanic.head())

# Check for missing values
print(titanic.isnull().sum())

print(titanic.columns)
# Handle missing values (e.g., impute age with median)
titanic['Age'].fillna(titanic['Age'].median(), inplace=True)

# Encode categorical variables (e.g., convert 'sex' to numeric)
titanic['Sex'] = titanic['Sex'].map({'male': 0, 'female': 1})

# Split data into features (X) and target variable (y)
X = titanic[['Sex','Age']]
y = titanic['Survived']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression

# Create an instance of logistic regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

a#ccuracy = accuracy_score(y_test, y_pred)
#print("Accuracy:", accuracy)

# Other evaluation metrics
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
