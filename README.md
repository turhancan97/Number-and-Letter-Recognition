# Number and Letter Recognition

## Açıklama
Bu proje basit bir sayı ve harf tanıma programıdır. El yazısı sayıları ve harfleri tanımak için bir sinir ağı, SVM, Lojistik Regresyon ve KNN'yi eğitmek için MNIST ve Alfabe veri kümesini kullanır. Program Python'da yazılmıştır ve Machine Learning modelini oluşturmak için sklearn kütüphanesini kullanır. Program daha sonra kullanıcıdan bir pencerede bir sayı veya harf çizmesini veya bir resim yüklemesini ister. Program daha sonra kullanıcının hangi sayıyı veya harfi çizdiğini tahmin etmek için modeli kullanır.

## Kurulum

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