import pandas as pd
import numpy as np
from load_and_save_files import load_mnist, load_alphabet, save_pickle

# MNIST ve Alphanet veri setlerini yükleyin
train_x_mnist, train_y_mnist, test_x_mnist, test_y_mnist = load_mnist()
train_x_alphabet, train_y_alphabet, valid_x_alphabet, valid_y_alphabet, test_x_alphabet, test_y_alphabet = load_alphabet()

# Aplhabet veri setindeki eğitim ve doğrulama verilerini birleştirin çünkü bunlar eğitim verileri olarak kullanılacak
train_x_alphabet = np.concatenate((train_x_alphabet, valid_x_alphabet))
train_y_alphabet = np.concatenate((train_y_alphabet, valid_y_alphabet))

# Pandas DataFrame'lerine dönüştürün
train_x_mnist, train_y_mnist, test_x_mnist, test_y_mnist = [
    pd.DataFrame(x) for x in [train_x_mnist, train_y_mnist, test_x_mnist, test_y_mnist]
]
train_x_alphabet, train_y_alphabet, test_x_alphabet, test_y_alphabet = [
    pd.DataFrame(x)
    for x in [train_x_alphabet, train_y_alphabet, test_x_alphabet, test_y_alphabet]
]

# MNIST veri setindeki verileri karıştırın çünkü veriler sıralı olarak depolanmıştır
train_x_mnist, train_y_mnist = train_x_mnist.sample(
    frac=1, random_state=42
), train_y_mnist.sample(frac=1, random_state=42)
test_x_mnist, test_y_mnist = test_x_mnist.sample(
    frac=1, random_state=42
), test_y_mnist.sample(frac=1, random_state=42)
train_x_alphabet, train_y_alphabet = train_x_alphabet.sample(
    frac=1, random_state=42
), train_y_alphabet.sample(frac=1, random_state=42)
test_x_alphabet, test_y_alphabet = test_x_alphabet.sample(
    frac=1, random_state=42
), test_y_alphabet.sample(frac=1, random_state=42)

# MNIST veri setinde 60000 eğitim verisi var. Alphanet veri setinde yakaşık 15000 eğitim verisi var.
# Bu şekilde eğitirsek MNIST veri seti çok daha fazla ağırlığa sahip olacak. Ve bu da modelin MNIST veri setindeki rakamları daha iyi tahmin etmesine neden olacak.
# Bu nedenle MNIST veri setindeki verilerin %10'unu kullanacağız.
# Aynı şekilde test verileri için de yapacağız. MNIST veri setindeki verilerin %7.5'unu kullanacağız.
# Böylelikle rakamlar ve harfler için eğitim ve test verileri benzer dağılıma sahip olacak.
train_x_mnist = train_x_mnist.iloc[: int(len(train_x_mnist) * 0.1)]
train_y_mnist = train_y_mnist.iloc[: int(len(train_y_mnist) * 0.1)]
train_x = pd.concat([train_x_alphabet, train_x_mnist])
train_y = pd.concat([train_y_alphabet, train_y_mnist])

test_x_mnist = test_x_mnist.iloc[: int(len(test_x_mnist) * 0.075)]
test_y_mnist = test_y_mnist.iloc[: int(len(test_y_mnist) * 0.075)]
test_x = pd.concat([test_x_alphabet, test_x_mnist])
test_y = pd.concat([test_y_alphabet, test_y_mnist])

print("*************************************")
print("Shape of train and test data:\n")
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
print("*************************************")

# Verileri bir sorun yaşamamak için tekrar karıştırın
train_x, train_y = train_x.sample(frac=1, random_state=42), train_y.sample(
    frac=1, random_state=42
)
test_x, test_y = test_x.sample(frac=1, random_state=42), test_y.sample(
    frac=1, random_state=42
)

# Verileri normalleştirin.
# Çünkü bu durum modelin hem daha hızlı eğitilmesini hem de daha iyi sonuçlar vermesini sağlar.
train_x = train_x / 255.0
test_x = test_x / 255.0

# Preprocessed_data.pkl adıyla pickle olarak kaydedin
data = {
    "train_x": train_x,
    "train_y": train_y,
    "test_x": test_x,
    "test_y": test_y,
}
save_pickle(data, "data/preprocessed_data.pkl")
