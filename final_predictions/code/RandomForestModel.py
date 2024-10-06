from sklearn.ensemble import RandomForestClassifier
import numpy as np

class RandomForestModel:
    def __init__(self):
        self.model = None

    def train_random_forest(self, training_data, training_labels):
        """
        Trains a Random Forest classifier on the provided data and labels.

        :param training_data: List of lists containing feature data (each inner list should contain 5 fields).
        :param training_labels: List of boolean values indicating the target variable (True/False).
        :return: None.
        """
        # Convert data and labels to numpy arrays
        X = np.array(training_data)
        y = np.array(training_labels)

        # Ensure that the feature set has exactly 5 fields per example
        if X.shape[1] != 5:
            raise ValueError("Each inner list in 'data' must contain exactly 5 fields.")

        # Creating a Random Forest classifier
        clf = RandomForestClassifier(n_estimators=100)

        # Training the model on the training dataset
        clf.fit(X, y)

        # Save the trained model in the class
        self.model = clf

    def predict(self, data):
        """
        Makes a prediction for a single data point after the model has been trained.

        :param data: A single list of features (must contain exactly 5 fields).
        :return: True or False based on the model's prediction.
        """
        if self.model is None:
            raise ValueError("The model has not been trained yet. Please train it first.")
        
        # Convert the input data to numpy array
        X_test = np.array(data).reshape(1, -1)  # Reshape to make it 2D array with 1 row

        # Ensure that the test data has exactly 5 fields
        if X_test.shape[1] != 5:
            raise ValueError("Input data must contain exactly 5 fields.")

        # Perform the prediction
        y_pred = self.model.predict(X_test)

        # Return the predicted boolean value
        return bool(y_pred[0])
