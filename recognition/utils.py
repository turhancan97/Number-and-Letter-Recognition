import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score


def adjust_grayscale(image, threshold=65):
    """
    Adjusts the grayscale values of an image by applying a threshold.
    """
    return image.point(lambda x: 0 if x < threshold else x)


def apply_erosion(image_array, kernel_size=(3, 3), iterations=1):
    """
    Performs erosion on the input image sequence using the specified kernel size and iterations.
    """
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.erode(image_array, kernel, iterations=iterations)


def train_model(
    model_name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
):
    """
    It trains a machine learning model based on the given model name and input data.
    """
    if model_name == "K Nearest Neighbors":
        model = KNeighborsClassifier()
    elif model_name == "Logistic Regression":
        model = LogisticRegression(verbose=True, random_state=42)
    elif model_name == "Multi Layer Perceptron":
        model = MLPClassifier(verbose=True, random_state=42)
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(verbose=True, random_state=42)
    model.fit(train_x, train_y.values.flatten())
    y_pred = model.predict(test_x)
    test_acc = accuracy_score(test_y, y_pred) * 100
    print(classification_report(test_y, y_pred))

    return model, test_acc
