import sys
import numpy as np
import pickle
from PIL import Image, ImageOps, ImageQt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QSize

from recognition.utils import apply_erosion, adjust_grayscale

class MainWidget(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()

        # Canvas boyutu
        self.QPixmap_size = 256, 256

        # Eğitim verileri boyutu
        self.training_size = 28, 28

        # Bir görüntünün diskten yüklenip yüklenmediğini kontrol etmek için bayrak
        self.image_loaded_from_disk = False

        # Fırça boyutu
        self.brush_size = 5

        # En iyi modeli yükleme
        self.loaded_model = pickle.load(open('model/best_model.pkl', 'rb'))

        # Arayüzü oluşturma
        self.initUI()

    def initUI(self):
        self.container = QtWidgets.QVBoxLayout()
        self.container.setContentsMargins(0, 0, 0, 0)

        # 'docs' dizininde simgeleriniz var
        clear_icon = QIcon('docs/clean.png')
        predict_icon = QIcon('docs/predict.png')
        load_icon = QIcon('docs/load.png')

        # Kanvas Konteyner Olarak Kullanılır
        self.label = QtWidgets.QLabel()
        canvas = QtGui.QPixmap(self.QPixmap_size[0], self.QPixmap_size[1])
        canvas.fill(QtGui.QColor("black"))
        self.label.setPixmap(canvas)
        self.last_x, self.last_y = None, None

        # Tahmin etme sonucunu göstermek için bir etiket
        self.prediction = QtWidgets.QLabel('Prediction: ...')
        self.prediction.setFont(QtGui.QFont('Arial', 20, QtGui.QFont.Bold))
        self.prediction.setStyleSheet("color: blue;")

        # Fırça boyutunu ayarlamak için bir kaydırıcı
        self.slider_brush_size = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.slider_brush_size.setMinimum(1)
        self.slider_brush_size.setMaximum(20)
        self.slider_brush_size.setValue(self.brush_size)
        self.slider_brush_size.valueChanged[int].connect(self.change_brush_size)

        # Temizleme, tahmin etme ve görüntü yükleme düğmeleri
        self.button_clear = QtWidgets.QPushButton('CLEAR')
        self.button_clear.setIcon(clear_icon)
        self.button_clear.setIconSize(QSize(24, 24))  # Adjust icon size
        self.button_clear.setStyleSheet("background-color: red; color: white;")
        self.button_clear.clicked.connect(self.clear_canvas)

        # Tahmin etme düğmesi
        self.button_save = QtWidgets.QPushButton('PREDICT')
        self.button_save.setIcon(predict_icon)
        self.button_save.setIconSize(QSize(24, 24))
        self.button_save.setStyleSheet("background-color: green; color: white;")
        self.button_save.clicked.connect(self.predict)

        # Görüntü yükleme düğmesi
        self.button_load_image = QtWidgets.QPushButton('LOAD IMAGE')
        self.button_load_image.setIcon(load_icon)
        self.button_load_image.setIconSize(QSize(24, 24))
        self.button_load_image.setStyleSheet("background-color: orange; color: white;")
        self.button_load_image.clicked.connect(self.load_image_from_disk)

        # Konteyneri düğmelerle doldurun
        self.container.addWidget(self.label)
        self.container.addWidget(self.prediction, alignment=QtCore.Qt.AlignHCenter)
        self.container.addWidget(self.button_clear)
        self.container.addWidget(self.button_save)
        self.container.addWidget(self.button_load_image)
        brush_size_label = QtWidgets.QLabel("Brush Size")
        brush_size_label.setStyleSheet("color: blue;")
        brush_size_label.setFont(QtGui.QFont('Arial', 15, QtGui.QFont.Bold))
        self.container.addWidget(brush_size_label, alignment=QtCore.Qt.AlignHCenter)
        self.container.addWidget(self.slider_brush_size)

        self.setLayout(self.container)

    def change_brush_size(self, value):
        '''
        Fırça boyutunu değiştirmek için kullanılır.
        '''
        self.brush_size = value

    def load_image_from_disk(self):
        '''
        Diskten bir görüntü yüklemek için kullanılır.
        '''
        options = QtWidgets.QFileDialog.Options()
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Image", "", "Image Files (*.png *.jpg *.jpeg)", options=options
        )
        if filename:
            image = Image.open(filename)
            image = image.resize(self.QPixmap_size, Image.ANTIALIAS)
            image_qt = ImageQt.ImageQt(image)  # Convert PIL image to QImage
            pixmap = QtGui.QPixmap.fromImage(image_qt)
            self.label.setPixmap(pixmap)
            self.image_loaded_from_disk = True  # Set the flag when an image is loaded

    def clear_canvas(self):
        '''
        Kanvası temizlemek için kullanılır.
        '''
        self.label.pixmap().fill(QtGui.QColor('#000000'))
        self.update()

    def predict(self):
        '''
        Kanvas üzerindeki görüntüyü tahmin etmek için kullanılır.
        '''
        s = self.label.pixmap().toImage().bits().asarray(self.QPixmap_size[0] * self.QPixmap_size[1] * 4)
        arr = np.frombuffer(s, dtype=np.uint8).reshape((self.QPixmap_size[0], self.QPixmap_size[1], 4))
        if self.image_loaded_from_disk:  # Eğer görüntü diskten yüklendiyse
            img = Image.fromarray(arr)
            img_gray = ImageOps.grayscale(img)
            # Ortalama pizel değeri 128'den fazlaysa görüntüyü ters çevirin
            # Bunun nedeni içeri aktarılan görüntülerin beyaz arka plana sahip olma ihtimalidir.
            # Bu nedenle görüntüyü ters çevirerek siyah arka plana sahip hale getiriyoruz.
            # Çünkü eğittiğimiz model siyah arka plana sahip görüntülerle eğitildi.
            if np.mean(img_gray) > 128:  # 128, 0 ve 255'in ortasıdır
                img_gray = ImageOps.invert(img_gray)  # Görüntüyü ters çevirin
                img_gray = img_gray.resize((self.training_size[0], self.training_size[1]), Image.ANTIALIAS)
                img_gray = adjust_grayscale(img_gray)  # Adjust grayscale
            else:
                img_gray = img_gray.resize((self.training_size[0], self.training_size[1]), Image.ANTIALIAS)
                arr = np.array(img_gray)
                arr = apply_erosion(arr, kernel_size=(2, 2), iterations=1)  # Apply erosion
                img_gray = Image.fromarray(arr)
        else:
            img = Image.fromarray(arr)
            img = img.resize((self.training_size[0], self.training_size[1]), Image.ANTIALIAS)
            img_gray = ImageOps.grayscale(img)
        arr = np.array(img_gray)
        arr = (arr / 255.0).reshape(1, -1)
        if self.loaded_model.predict(arr)[0] < 10:  # Eğer tahmin edilen değer 10'dan küçükse yani rakamsa
            prediction_value = str(self.loaded_model.predict(arr)[0])
        else:  # Eğer tahmin edilen değer 10'dan büyükse yani harfse
            prediction_value = chr(self.loaded_model.predict(arr)[0])
        self.prediction.setText('Prediction: ' + prediction_value)
        self.image_loaded_from_disk = False

    def mouseMoveEvent(self, e):
        '''
        Kanvas üzerindeki fare hareketini izlemek için kullanılır.
        '''
        if self.last_x is None:  # İlk etkinlik.
            self.last_x = e.x()
            self.last_y = e.y()
            return  # İlk seferi görmezden gel.

        painter = QtGui.QPainter(self.label.pixmap())

        p = painter.pen()
        p.setWidth(self.brush_size)  # Güncellenmiş fırça boyutunu kullanın
        self.pen_color = QtGui.QColor('#FFFFFF')  # Beyaz renk kullanın
        p.setColor(self.pen_color)
        painter.setPen(p)

        painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
        painter.end()
        self.update()

        # Update the origin for next time.
        self.last_x = e.x()
        self.last_y = e.y()

    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None

class MainWindow(QtWidgets.QMainWindow):
    '''
    Ana pencereyi oluşturmak için kullanılır.
    '''
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.mainWidget = MainWidget()
        self.setCentralWidget(self.mainWidget)


if __name__ == "__main__":
    '''
    Uygulamayı çalıştırmak için kullanılır.
    '''
    app = QtWidgets.QApplication([])
    mainApp = MainWindow()
    mainApp.setWindowTitle('Number&Letter Recognition')
    mainApp.show()
    sys.exit(app.exec_())
