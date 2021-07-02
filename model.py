# Import the libraries
import pandas as pd
import numpy as np

# Import the dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, [4, 5, 2]].values
y = dataset.iloc[:, 1].values

# Take care of missing data: age
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, [1]])
X[:, [1]] = imputer.transform(X[:, [1]])

# Encode categorical data: gender
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Split the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train model: logistic regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Confusion matrix and accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))

# Save the model
import joblib
joblib.dump(classifier, 'model.pkl')
print("Model sucesfully dumped")

# Save standard scaler
joblib.dump(sc, 'sc.pkl')
print("Standard scaler sucesfully dumped")
