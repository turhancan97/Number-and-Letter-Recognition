import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score


def adjust_grayscale(image, threshold=65):
    """
    Adjusts the grayscale values of an image by applying a threshold.

    Args:
        image (PIL.Image.Image): The input image.
        threshold (int): The threshold value. Pixels with values below this threshold will be set to 0.

    Returns:
        PIL.Image.Image: The adjusted image.
    """
    return image.point(lambda x: 0 if x < threshold else x)


def apply_erosion(image_array, kernel_size=(3, 3), iterations=1):
    """
    Applies erosion operation on the input image array using the specified kernel size and iterations.

    Parameters:
        image_array (numpy.ndarray): The input image array.
        kernel_size (tuple): The size of the kernel used for erosion. Default is (3, 3).
        iterations (int): The number of times erosion is applied. Default is 1.

    Returns:
        numpy.ndarray: The eroded image array.
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
    Trains a machine learning model based on the given model name and input data.

    Args:
        model_name (str): The name of the model to train.
        train_x (np.ndarray): The input features for training.
        train_y (np.ndarray): The target labels for training.
        test_x (np.ndarray): The input features for testing.
        test_y (np.ndarray): The target labels for testing.

    Returns:
        The trained machine learning model.
    """
    if model_name == "Support Vector Machine":
        model = SVC(verbose=True, random_state=42)
    elif model_name == "K Nearest Neighbors":
        model = KNeighborsClassifier()
    elif model_name == "Logistic Regression":
        model = LogisticRegression(verbose=True, random_state=42)
    elif model_name == "Multi Layer Perceptron":
        model = MLPClassifier(verbose=True, random_state=42)
    model.fit(train_x, train_y.values.flatten())
    y_pred = model.predict(test_x)
    test_acc = accuracy_score(test_y, y_pred) * 100
    print(classification_report(test_y, y_pred))

    return model, test_acc
