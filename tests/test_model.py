import unittest
from src.model import create_model
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class TestModel(unittest.TestCase):
    def test_model_accuracy(self):
        data = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            data.data, data.target, test_size=0.2, random_state=42
        )
        model = create_model()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        self.assertGreater(accuracy, 0.7)


if __name__ == "__main__":
    unittest.main()
