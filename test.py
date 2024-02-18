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

        # Canvas size
        self.QPixmap_size = 256, 256

        # Training data size
        self.training_size = 28, 28

        # Flag to check if an image is loaded from disk
        self.image_loaded_from_disk = False

        # Brush size
        self.brush_size = 5

        # Load the best model
        self.loaded_model = pickle.load(open("model/best_model.pkl", "rb"))

        # Create the interface
        self.initUI()

    def initUI(self):
        self.container = QtWidgets.QVBoxLayout()
        self.container.setContentsMargins(0, 0, 0, 0)

        # We have icons in the 'docs' directory
        clear_icon = QIcon("docs/clean.png")
        predict_icon = QIcon("docs/predict.png")
        load_icon = QIcon("docs/load.png")

        # Canvas Used as Container
        self.label = QtWidgets.QLabel()
        canvas = QtGui.QPixmap(self.QPixmap_size[0], self.QPixmap_size[1])
        canvas.fill(QtGui.QColor("black"))
        self.label.setPixmap(canvas)
        self.last_x, self.last_y = None, None

        # A label to show the prediction result
        self.prediction = QtWidgets.QLabel("Prediction: ...")
        self.prediction.setFont(QtGui.QFont("Arial", 20, QtGui.QFont.Bold))
        self.prediction.setStyleSheet("color: blue;")

        # A slider to adjust the brush size
        self.slider_brush_size = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.slider_brush_size.setMinimum(1)
        self.slider_brush_size.setMaximum(20)
        self.slider_brush_size.setValue(self.brush_size)
        self.slider_brush_size.valueChanged[int].connect(self.change_brush_size)

        # Clear, predict and image upload buttons
        self.button_clear = QtWidgets.QPushButton("CLEAR")
        self.button_clear.setIcon(clear_icon)
        self.button_clear.setIconSize(QSize(24, 24))  # Adjust icon size
        self.button_clear.setStyleSheet("background-color: red; color: white;")
        self.button_clear.clicked.connect(self.clear_canvas)

        # Prediction button
        self.button_save = QtWidgets.QPushButton("PREDICT")
        self.button_save.setIcon(predict_icon)
        self.button_save.setIconSize(QSize(24, 24))
        self.button_save.setStyleSheet("background-color: green; color: white;")
        self.button_save.clicked.connect(self.predict)

        # Image upload button
        self.button_load_image = QtWidgets.QPushButton("LOAD IMAGE")
        self.button_load_image.setIcon(load_icon)
        self.button_load_image.setIconSize(QSize(24, 24))
        self.button_load_image.setStyleSheet("background-color: orange; color: white;")
        self.button_load_image.clicked.connect(self.load_image_from_disk)

        # Fill the container with buttons
        self.container.addWidget(self.label)
        self.container.addWidget(self.prediction, alignment=QtCore.Qt.AlignHCenter)
        self.container.addWidget(self.button_clear)
        self.container.addWidget(self.button_save)
        self.container.addWidget(self.button_load_image)
        brush_size_label = QtWidgets.QLabel("Brush Size")
        brush_size_label.setStyleSheet("color: blue;")
        brush_size_label.setFont(QtGui.QFont("Arial", 15, QtGui.QFont.Bold))
        self.container.addWidget(brush_size_label, alignment=QtCore.Qt.AlignHCenter)
        self.container.addWidget(self.slider_brush_size)

        self.setLayout(self.container)

    def change_brush_size(self, value):
        """
        Used to change the brush size.
        """
        self.brush_size = value

    def load_image_from_disk(self):
        """
        Used to load an image from disk.
        """
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
        """
        Used for cleaning the canvas.
        """
        self.label.pixmap().fill(QtGui.QColor("#000000"))
        self.update()

    def predict(self):
        """
        Used to estimate the image on the canvas.
        """
        s = (
            self.label.pixmap()
            .toImage()
            .bits()
            .asarray(self.QPixmap_size[0] * self.QPixmap_size[1] * 4)
        )
        arr = np.frombuffer(s, dtype=np.uint8).reshape(
            (self.QPixmap_size[0], self.QPixmap_size[1], 4)
        )
        if self.image_loaded_from_disk:  # If the image was loaded from disk
            img = Image.fromarray(arr)
            img_gray = ImageOps.grayscale(img)
            # If the average pixel value is greater than 128, invert the image
            # This is because the imported images may have a white background.
            # Therefore, we invert the image to have a black background.
            # The trained model was trained with images having a black background.
            if np.mean(img_gray) > 128:  # 128 is the middle value between 0 and 255
                img_gray = ImageOps.invert(img_gray)
                img_gray = img_gray.resize(
                    (self.training_size[0], self.training_size[1]), Image.ANTIALIAS
                )
                img_gray = adjust_grayscale(img_gray)  # Adjust grayscale
            else:
                img_gray = img_gray.resize(
                    (self.training_size[0], self.training_size[1]), Image.ANTIALIAS
                )
                arr = np.array(img_gray)
                arr = apply_erosion(
                    arr, kernel_size=(2, 2), iterations=1
                )  # Apply erosion
                img_gray = Image.fromarray(arr)
        else:
            img = Image.fromarray(arr)
            img = img.resize(
                (self.training_size[0], self.training_size[1]), Image.ANTIALIAS
            )
            img_gray = ImageOps.grayscale(img)
        arr = np.array(img_gray)
        arr = (arr / 255.0).reshape(1, -1)
        if (
            self.loaded_model.predict(arr)[0] < 10
        ):  # If the estimated value is less than 10, i.e. a number
            prediction_value = str(self.loaded_model.predict(arr)[0])
        else:  # If the predicted value is greater than 10, i.e. a letter
            prediction_value = chr(self.loaded_model.predict(arr)[0])
        self.prediction.setText("Prediction: " + prediction_value)
        self.image_loaded_from_disk = False

    def mouseMoveEvent(self, e):
        """
        Used to track mouse movement on the canvas.
        """
        if self.last_x is None:  # First event
            self.last_x = e.x()
            self.last_y = e.y()
            return  # Ignore the first time.

        painter = QtGui.QPainter(self.label.pixmap())

        p = painter.pen()
        p.setWidth(self.brush_size)  # Use the updated brush size
        self.pen_color = QtGui.QColor("#FFFFFF")  # Use white color
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
    """
    Used to create the main window.
    """

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.mainWidget = MainWidget()
        self.setCentralWidget(self.mainWidget)


if __name__ == "__main__":
    """
    Used to run the application.
    """
    app = QtWidgets.QApplication([])
    mainApp = MainWindow()
    mainApp.setWindowTitle("Number&Letter Recognition")
    mainApp.show()
    sys.exit(app.exec_())
