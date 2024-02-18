import os
from PIL import Image, ImageOps
import numpy as np
import pickle
from recognition.utils import adjust_grayscale

def from_jpg_to_png(folder):
    """
    Converts all JPG images in the specified folder and its subfolders to PNG format.
    This is because the alphanet dataset is in JPG format and it is easier to work with PNG format.
    """
    for label in os.listdir(folder):
        label_path = os.path.join(folder, label)
        if os.path.isdir(label_path):
            for image_file in os.listdir(label_path):
                if image_file.endswith('.jpg'):
                    print(label_path, end='\r')
                    img_path = os.path.join(label_path, image_file)
                    img = Image.open(img_path)
                    img.save(img_path.replace('.jpg', '.png'))
                    os.remove(img_path)

def load_images_from_folder(folder):
    """
    Load images from a specified folder and convert them to a numpy array.
    Each image in the Alphanet dataset is 256x256 pixels in size.
    We then resize these images to 28x28 pixels in size.
    This is because smaller images train faster.
    Also, using the same size as the MNIST dataset makes it easier to combine the training and test data later on.
    We also grayscale the images. This is done to make it easier to process the images later.
    And we grayscale them to make them in the same form as the MNIST dataset.
    """
    images = []
    labels = []
    for label in os.listdir(folder): # Iterate through all folders
        label_path = os.path.join(folder, label) # Create folder path
        if os.path.isdir(label_path): # If it is a folder
            for image_file in os.listdir(label_path): # Iterate through all files in the folder
                if image_file.endswith('.png'): # If the file extension is .png
                    print(label_path, end='\r') # Print the process
                    img_path = os.path.join(label_path, image_file) # Create file path
                    img = Image.open(img_path).convert('L')  # Convert to grayscale
                    img = ImageOps.invert(img)  # Convert to grayscale with black background
                    img = img.resize((28, 28), Image.ANTIALIAS)  # Resize with a higher quality downsampling filter
                    img = adjust_grayscale(img)  # Adjust grayscale
                    img_array = np.array(img).astype(int).flatten() # Convert to array
                    images.append(img_array) # Append the array to the image list
                    labels.append(ord(label))  # Convert the label to a numerical value
    return np.array(images), np.array(labels)

def process_and_save_data(training_dir, validation_dir, testing_dir, output_file):
    """
    Process the data in the given directories and save it in a pickle file.
    """
    training_images, training_labels = load_images_from_folder(training_dir) # Load training data
    validation_images, validation_labels = load_images_from_folder(validation_dir) # Load validation data
    test_images, test_labels = load_images_from_folder(testing_dir) # Load test data

    data = {
        'training_images': training_images,
        'training_labels': training_labels,
        'validation_images': validation_images,
        'validation_labels': validation_labels,
        'test_images': test_images,
        'test_labels': test_labels
    } # Save the data in a dictionary

    with open(output_file, 'wb') as f:
        pickle.dump(data, f) # Save the data to a pickle file

# Folders of the data set
train_folder = './data/alphabet/train'
validation_folder = './data/alphabet/validation'
test_folder = './data/alphabet/test'

# Change to png
for folder in [train_folder, validation_folder, test_folder]:
    from_jpg_to_png(folder)

# Processing and saving data
process_and_save_data(train_folder, validation_folder, test_folder, 'data/alphabet.pkl')
