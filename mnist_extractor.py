import os
from PIL import Image
import numpy as np
import pickle
from utils import apply_erosion

def load_images_from_folder(folder):
    images = []
    labels = []
    for label in os.listdir(folder):
        label_path = os.path.join(folder, label)
        if os.path.isdir(label_path):
            for image_file in os.listdir(label_path):
                if image_file.endswith('.png'):
                    print(label_path, end='\r')
                    img_path = os.path.join(label_path, image_file)
                    img = Image.open(img_path)
                    # img = img.resize((28, 28))  # Resize to 28x28
                    img_array = np.array(img)
                    img_array = apply_erosion(img_array, kernel_size=(2, 2), iterations=1)  # Apply erosion
                    img_array = img_array.flatten()  # Flatten the image
                    images.append(img_array)
                    labels.append(int(label))
    return np.array(images), np.array(labels)

def process_and_save_mnist(training_dir, testing_dir, output_file):
    training_images, training_labels = load_images_from_folder(training_dir)
    test_images, test_labels = load_images_from_folder(testing_dir)
    
    mnist = {
        'training_images': training_images,
        'training_labels': training_labels,
        'test_images': test_images,
        'test_labels': test_labels
    }

    with open(output_file, 'wb') as f:
        pickle.dump(mnist, f)

# Example usage
process_and_save_mnist('./mnist/training', './mnist/testing', 'mnist.pkl')