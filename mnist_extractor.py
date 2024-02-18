import os
from PIL import Image
import numpy as np
import pickle
from recognition.utils import apply_erosion

def load_images_from_folder(folder):
    """
    Belirtilen bir klasörden görüntüleri yükleyin ve bunları bir numpy dizisine dönüştürün.
    MNIST veri setindeki her görüntü 28x28 piksel boyutundadır.
    Verileri erozyon uygulayarak işliyoruz.
    Bunun nedeni, Alphabet verisetindeki görüntülerin MNIST veri setindeki görüntülerden daha ince olmasıdır.
    Böylelikle MNSIT veri setindeki rakamları inceltiyoruz ve Alphabet veri setindeki görüntülerle benzer dağılımsal özelliklere sahip oluyoruz.
    """
    images = []
    labels = []
    for label in os.listdir(folder): # Tüm klasörleri dolaş
        label_path = os.path.join(folder, label) # Klasör yolunu oluştur
        if os.path.isdir(label_path): # Eğer klasör ise
            for image_file in os.listdir(label_path): # Klasördeki tüm dosyaları dolaş
                if image_file.endswith('.png'): # Eğer dosya uzantısı .png ise
                    print(label_path, end='\r') # İşlemi ekrana yaz
                    img_path = os.path.join(label_path, image_file) # Dosya yolunu oluştur
                    img = Image.open(img_path) # Görüntüyü aç
                    # img = img.resize((28, 28))  # Resize to 28x28
                    img_array = np.array(img) # numpy dizisine dönüştür
                    img_array = apply_erosion(img_array, kernel_size=(2, 2), iterations=1)  # Apply erosion
                    img_array = img_array.flatten()  # Flatten the image
                    images.append(img_array) # Diziyi görüntü dizisine ekleme
                    labels.append(int(label)) # Etiketi sayısal değere dönüştürme
    return np.array(images), np.array(labels)

def process_and_save_mnist(training_dir, testing_dir, output_file):
    """
    Verilen dizinlerdeki verileri işleyin ve bir pickle dosyasına kaydedin.
    """
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

# Verileri işleme ve kaydetme
process_and_save_mnist('./data/mnist/training', './data/mnist/testing', 'data/mnist.pkl')