import os
from PIL import Image, ImageOps
import numpy as np
import pickle
from utils import adjust_grayscale

def from_jpg_to_png(folder):
    """
    Belirtilen klasör ve alt klasörlerindeki tüm JPG görüntülerini PNG formatına dönüştürür.
    Bunun nedeni alphanet veri setinin JPG formatında olması ve PNG formatıyla çalışmanın daha kolay olmasıdır.
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
    Belirtilen bir klasörden görüntüleri yükleyin ve bunları bir numpy dizisine dönüştürün.
    Alphanet veri setindeki her görüntü 256x256 piksel boyutundadır.
    Daha sonra bu görüntüleri 28x28 piksel boyutuna yeniden boyutlandırıyoruz.
    Bunun nedeni, daha küçük görüntülerin daha hızlı eğitilmesidir.
    Ayrıca MNIST veri seti ile aynı boyutu kullanmak, daha sonra eğitim ve test verilerini birleştirmeyi kolaylaştırır.
    Ayrıca görüntüleri gri tonlamalı hale getiriyoruz. Bu, daha sonra görüntüleri daha kolay işlemek için yapılır.
    Ve MNIST veri seti ile aynı formtta yapmak için gri tonlamalı hale getiriyoruz.
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
                    # Siyah arka plan ile gri tonlamaya dönüştürme
                    img = Image.open(img_path).convert('L')  # Gri tonlamaya dönüştürme
                    img = ImageOps.invert(img)  # Siyah arka plan ile gri tonlamaya dönüştürme
                    img = img.resize((28, 28), Image.ANTIALIAS)  # Daha yüksek kaliteli bir alt örnekleme filtresi ile yeniden boyutlandırma
                    img = adjust_grayscale(img)  # Adjust grayscale
                    img_array = np.array(img).astype(int).flatten() # Diziye dönüştürme
                    images.append(img_array) # Diziyi görüntü dizisine ekleme
                    labels.append(ord(label))  # Etiketi sayısal değere dönüştürme
    return np.array(images), np.array(labels)

def process_and_save_data(training_dir, validation_dir, testing_dir, output_file):
    """
    Verilen dizinlerdeki verileri işleyin ve bir pickle dosyasına kaydedin.
    """
    training_images, training_labels = load_images_from_folder(training_dir) # Eğitim verilerini yükleme
    validation_images, validation_labels = load_images_from_folder(validation_dir) # Doğrulama verilerini yükleme
    test_images, test_labels = load_images_from_folder(testing_dir) # Test verilerini yükleme

    data = {
        'training_images': training_images,
        'training_labels': training_labels,
        'validation_images': validation_images,
        'validation_labels': validation_labels,
        'test_images': test_images,
        'test_labels': test_labels
    } # Verileri bir sözlüğe kaydetme

    with open(output_file, 'wb') as f:
        pickle.dump(data, f) # Verileri pickle dosyasına kaydetme

# Veri setinin klasörleri
train_folder = './data/alphabet/train'
validation_folder = './data/alphabet/validation'
test_folder = './data/alphabet/test'

# png olarak değiştir
for folder in [train_folder, validation_folder, test_folder]:
    from_jpg_to_png(folder)

# Verileri işleme ve kaydetme
process_and_save_data(train_folder, validation_folder, test_folder, 'data/alphabet.pkl')