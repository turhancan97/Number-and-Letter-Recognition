import sys
import numpy as np
import pickle
from PIL import Image, ImageOps, ImageQt
from PyQt5 import QtCore, QtGui, QtWidgets

from utils import apply_erosion

class MainWidget(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()

        # Canvas Size
        self.QPixmap_size = 256, 256

        # Training Size
        self.training_size = 28, 28

        self.image_loaded_from_disk = False  # Initialize the flag

        # Best Model Loaded
        self.loaded_model = pickle.load(open('multi_layer_perceptron.pkl', 'rb'))

        self.initUI()
    
    def initUI(self):
        self.container = QtWidgets.QVBoxLayout()
        self.container.setContentsMargins(0, 0, 0, 0)

        # Used As Canvas Container
        self.label = QtWidgets.QLabel()
        canvas = QtGui.QPixmap(self.QPixmap_size[0], self.QPixmap_size[1])
        canvas.fill(QtGui.QColor("black"))
        self.label.setPixmap(canvas)
        self.last_x, self.last_y = None, None

        self.prediction = QtWidgets.QLabel('Prediction: ...')
        self.prediction.setFont(QtGui.QFont('Monospace', 20))

        self.button_clear = QtWidgets.QPushButton('CLEAR')
        self.button_clear.clicked.connect(self.clear_canvas)

        self.button_save = QtWidgets.QPushButton('PREDICT')
        self.button_save.clicked.connect(self.predict)

        self.button_load_image = QtWidgets.QPushButton('LOAD IMAGE')
        self.button_load_image.clicked.connect(self.load_image_from_disk)

        self.container.addWidget(self.label)
        self.container.addWidget(self.prediction, alignment = QtCore.Qt.AlignHCenter)
        self.container.addWidget(self.button_clear)
        self.container.addWidget(self.button_save)
        self.container.addWidget(self.button_load_image)

        self.setLayout(self.container)

    def load_image_from_disk(self):
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
        self.label.pixmap().fill(QtGui.QColor('#000000'))
        self.update()

    def predict(self):
        s = self.label.pixmap().toImage().bits().asarray(self.QPixmap_size[0] * self.QPixmap_size[1] * 4)
        arr = np.frombuffer(s, dtype=np.uint8).reshape((self.QPixmap_size[0], self.QPixmap_size[1], 4))
        if self.image_loaded_from_disk:
            img = Image.fromarray(arr)
            img_gray = ImageOps.grayscale(img)
            # If average pizel value is more than 128, invert the image
            if np.mean(img_gray) > 128: # 128 is the middle of 0 and 255
                img_gray = ImageOps.invert(img_gray) # Invert the image
                img_gray = img_gray.resize((self.training_size[0], self.training_size[1]), Image.ANTIALIAS)
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
        if self.loaded_model.predict(arr)[0] < 10:
            prediction_value = str(self.loaded_model.predict(arr)[0])
        else:
            prediction_value = chr(self.loaded_model.predict(arr)[0])
        self.prediction.setText('Prediction: ' + prediction_value)
        self.image_loaded_from_disk = False
    
    def mouseMoveEvent(self, e):
        if self.last_x is None: # First event.
            self.last_x = e.x()
            self.last_y = e.y()
            return # Ignore the first time.

        painter = QtGui.QPainter(self.label.pixmap())

        p = painter.pen()
        p.setWidth(5)
        self.pen_color = QtGui.QColor('#FFFFFF')
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

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.mainWidget = MainWidget()
        self.setCentralWidget(self.mainWidget)



if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    mainApp = MainWindow()
    mainApp.setWindowTitle('DIGIT PREDICTER')
    mainApp.show()
    sys.exit(app.exec_())