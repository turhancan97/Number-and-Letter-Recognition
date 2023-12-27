# Number and Letter Recognition

## Açıklama

Bu proje basit bir sayı ve harf tanıma programıdır. El yazısı sayıları ve harfleri tanımak için bir sinir ağı, Decision Tree, Random Forest, Lojistik Regresyon ve KNN'yi eğitmek için MNIST ve Alfabe veri kümesini kullanır. Program Python'da yazılmıştır ve Machine Learning modelini oluşturmak için sklearn kütüphanesini kullanır. Program daha sonra kullanıcıdan bir pencerede bir sayı veya harf çizmesini veya bir resim yüklemesini ister. Program daha sonra kullanıcının hangi sayıyı veya harfi çizdiğini tahmin etmek için modeli kullanır.

## Kurulum

```bash
conda create -n number_letter_recognition python=3.9
conda activate number_letter_recognition
```

- Eğer conda kurulu değilse [buradan](https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html) indirebilirsiniz.
- Eğer environment kurmak için conda kullanmak istemiyorsanız venv kullanabilirsiniz. [Buradan](https://docs.python.org/3/library/venv.html) venv hakkında bilgi alabilirsiniz.
- Daha sonra gerekli kütüphaneleri aşağıdaki komut ile kurabilirsiniz.

```bash
pip install -r requirements.txt
```

## Veri klasörü yapısı

```bash
data/
├── alphabet/
│   ├── train/
│   │   ├── a
│   │   ├── b
│   │   ├── c
│   │   └── ...
│   ├── validation/
│   │   ├── a
│   │   ├── b
│   │   ├── c
│   │   └── ...
│   └── test/
│       ├── a
│       ├── b
│       ├── c
│       └── ..
└── mnist/
    ├── training
    ├── 0
    ├── 1
    ├── 2
    ├── ...
    ├── testing
    ├── 0
    ├── 1
    ├── 2
    └── ...
alphabet_extractor.py
mnist_extractor.py
```

### MNIST veri kümesini çıkarmak için

```bash
python mnist_extractor.py
```

- Mnist veri kümesi, 28x28 boyutunda 70.000 resim içerir. 60.000 eğitim resmi ve 10.000 test resmi vardır. Her resim, 0 ile 9 arasında bir sayıyı temsil eden bir etikete sahiptir.
- Verileri erozyon uygulayarak işliyoruz. Bunun nedeni, Alphabet verisetindeki görüntülerin MNIST veri setindeki görüntülerden daha ince olmasıdır. Böylelikle MNSIT veri setindeki rakamları inceltiyoruz ve Alphabet veri setindeki görüntülerle benzer dağılımsal özelliklere sahip oluyoruz.

![erozyon](docs/erozyon.png)

### Alfabe veri kümesini çıkarmak için

```bash
python alphabet_extractor.py
```

- Alphanet veri setindeki her görüntü 256x256 piksel boyutundadır.
- Daha sonra bu görüntüleri 28x28 piksel boyutuna yeniden boyutlandırıyoruz. Bunun nedeni, daha küçük görüntülerin daha hızlı eğitilmesidir. Ayrıca MNIST veri seti ile aynı boyutu kullanmak, daha sonra eğitim ve test verilerini birleştirmeyi kolaylaştırır.
- Bunlara ek olarak görüntüleri gri tonlamalı (arka siyah yazı beyaz) hale getiriyoruz. Bu, daha sonra görüntüleri daha kolay işlemek için ve MNIST veri seti ile aynı formata getirmek için yapılır

![düzeltme](docs/düzenleme.png)

## Veri önişleme

- Burada veriler birleştirilir ve daha sonra eğitim ve test verileri olarak bölünür. Ayrıca verileri normalleştiririz. Normalleştirme, verileri 0 ile 1 arasında bir değere dönüştürür. Bu, daha sonra eğitim verilerini daha hızlı eğitmek için yapılır.

```bash
python preprocessing.py
```

## Tüm Modelleri eğitmek için

```bash
python train.py
```

- Bu komut, eğitim verilerini kullanarak tüm modelleri eğitir ve daha sonra modelleri kaydeder. Ayrıca modelleri en iyiden en kötüye doğruluk sırasına göre sıralar.

## En iyi modeli eğitmek için

```bash
python best_model_train.py
```

## Grafikler

- Grafikler için result.ipynb dosyasına bakınız.

## Test ve Arayüz

```bash
python test.py
```

- Bu komut, kullanıcıdan bir pencerede bir sayı veya harf çizmesini veya bir resim yüklemesini ister. Program daha sonra kullanıcının hangi sayıyı veya harfi çizdiğini tahmin etmek için en iyi modeli kullanır.