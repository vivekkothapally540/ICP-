import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the glass dataset
glass_data = pd.read_csv("glass.csv")

# Split the data into features and target
X = glass_data.iloc[:, :-1].values
y = glass_data.iloc[:, -1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Train the GaussianNB model
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Make predictions on the test set
gnb_y_pred = gnb.predict(X_test)

# Evaluate the model performance
gnb_acc = accuracy_score(y_test, gnb_y_pred)
print("GaussianNB Accuracy: {:.2f}%".format(gnb_acc * 100))
print("\nNaïve Bayes Classification Report:")
print(classification_report(y_test, gnb_y_pred, zero_division=1))