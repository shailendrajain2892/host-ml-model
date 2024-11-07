# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix

import pickle


def load_iris_dataset():
    # Load the dataset from a URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

    # Read the dataset
    dataset = pd.read_csv(url, names=names)

    # Display the first 5 rows
    # print(dataset.head())

    # Separate features and target variable
    X = dataset.iloc[:, :-1].values  # Features (sepal-length, sepal-width, petal-length, petal-width)
    y = dataset.iloc[:, -1].values   # Target variable (Class)
    return X, y

def train_logistic_model(X, y):
    # Split data into training and test sets (80% training, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling (Standardization)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train a Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    return y_test, y_pred, model

def evaluate_model(y_test, y_pred):
    # Confusion matrix
    print(confusion_matrix(y_test, y_pred))

    # Classification report
    print(classification_report(y_test, y_pred))

def save_model(model):
    # Save the trained model to disk
    model_filename = "logistic_regression_iris.pkl"
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)

    print(f"Model saved to {model_filename}")

def main():
    x, y = load_iris_dataset()
    y_test, y_pred, model = train_logistic_model(x, y)
    evaluate_model(y_test, y_pred)
    save_model(model)

if __name__ == "__main__":
    main()  