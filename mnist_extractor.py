import os
from PIL import Image
import numpy as np
import pickle
from recognition.utils import apply_erosion


def load_images_from_folder(folder):
    """
    Load images from the specified folder and convert them into a numpy array.
    Each image in the MNIST dataset is of size 28x28 pixels.
    We apply erosion to the data.
    The reason for this is that the images in the Alphabet dataset are thinner than the images in the MNIST dataset.
    This way, we thin the numbers in the MNSIT dataset and have images in the Alphabet dataset with similar distributional characteristics.
    """
    images = []
    labels = []
    for label in os.listdir(folder):  # Iterate through all folders
        label_path = os.path.join(folder, label)  # Create folder path
        if os.path.isdir(label_path):  # If it is a folder
            for image_file in os.listdir(
                label_path
            ):  # Iterate through all files in the folder
                if image_file.endswith(".png"):  # If the file extension is .png
                    print(label_path, end="\r")  # Print the process to the screen
                    img_path = os.path.join(label_path, image_file)  # Create file path
                    img = Image.open(img_path)  # Open the image
                    # img = img.resize((28, 28))  # Resize to 28x28
                    img_array = np.array(img)  # Convert to numpy array
                    img_array = apply_erosion(
                        img_array, kernel_size=(2, 2), iterations=1
                    )  # Apply erosion
                    img_array = img_array.flatten()  # Flatten the image
                    images.append(img_array)  # Append the array to the image array
                    labels.append(int(label))  # Convert the label to a numerical value
    return np.array(images), np.array(labels)


def process_and_save_mnist(training_dir, testing_dir, output_file):
    """
    Process the data in the given directories and save it to a pickle file.
    """
    training_images, training_labels = load_images_from_folder(training_dir)
    test_images, test_labels = load_images_from_folder(testing_dir)

    mnist = {
        "training_images": training_images,
        "training_labels": training_labels,
        "test_images": test_images,
        "test_labels": test_labels,
    }

    with open(output_file, "wb") as f:
        pickle.dump(mnist, f)


# Process and save the data
process_and_save_mnist(
    "./data/mnist/training", "./data/mnist/testing", "data/mnist.pkl"
)
