import os
from PIL import Image, ImageOps
import numpy as np
import pickle
from utils import adjust_grayscale

def from_jpg_to_png(folder):
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
    images = []
    labels = []
    for label in os.listdir(folder):
        label_path = os.path.join(folder, label)
        if os.path.isdir(label_path):
            for image_file in os.listdir(label_path):
                if image_file.endswith('.png'):
                    print(label_path, end='\r')
                    img_path = os.path.join(label_path, image_file)
                    # Convert to grayscale with black background
                    img = Image.open(img_path).convert('L')
                    img = ImageOps.invert(img)
                    # img = img.filter(ImageFilter.FIND_EDGES)  # Apply edge detection
                    img = img.resize((28, 28), Image.ANTIALIAS)  # Resize with a higher-quality downsampling filter
                    img = adjust_grayscale(img)  # Adjust grayscale
                    # from True and False to 1 and 0 and Flatten the image
                    img_array = np.array(img).astype(int).flatten()
                    images.append(img_array)
                    labels.append(ord(label))  # Convert label to numerical
    return np.array(images), np.array(labels)

def process_and_save_data(training_dir, validation_dir, testing_dir, output_file):
    training_images, training_labels = load_images_from_folder(training_dir)
    validation_images, validation_labels = load_images_from_folder(validation_dir)
    test_images, test_labels = load_images_from_folder(testing_dir)
    
    data = {
        'training_images': training_images,
        'training_labels': training_labels,
        'validation_images': validation_images,
        'validation_labels': validation_labels,
        'test_images': test_images,
        'test_labels': test_labels
    }

    with open(output_file, 'wb') as f:
        pickle.dump(data, f)

# Example usage
train_folder = './data/alphabet/train'
validation_folder = './data/alphabet/validation'
test_folder = './data/alphabet/test'
# # change it to png
# for folder in [train_folder, validation_folder, test_folder]:
#     from_jpg_to_png(folder)
process_and_save_data(train_folder, validation_folder, test_folder, 'data/alphabet.pkl')