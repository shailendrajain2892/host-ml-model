import pickle

# Load the saved model from disk
with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)

# Use the loaded model to make predictions
loaded_y_pred = loaded_model.predict(X_test)

# Evaluate the loaded model
print("Loaded Model Evaluation:")
print(confusion_matrix(y_test, loaded_y_pred))
print(classification_report(y_test, loaded_y_pred))