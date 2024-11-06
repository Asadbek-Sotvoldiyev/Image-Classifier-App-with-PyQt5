import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QFileDialog, QVBoxLayout, QSpacerItem, QSizePolicy
from PyQt5.QtGui import QPixmap
from tensorflow import keras
import numpy as np
from PIL import Image


class ImageClassifierApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Image Classifier')
        self.setGeometry(750, 300, 400, 500)

        # Create buttons and labels
        self.upload_btn = QPushButton('Upload Image', self)
        self.upload_btn.clicked.connect(self.upload_image)
        self.upload_btn.setStyleSheet("font-size: 16px; padding: 10px;")

        self.predict_btn = QPushButton('Predict', self)
        self.predict_btn.clicked.connect(self.predict_image)
        self.predict_btn.setEnabled(False)
        self.predict_btn.setStyleSheet("font-size: 16px; padding: 10px;")

        self.image_label = QLabel(self)
        self.image_label.setFixedSize(200, 200)
        self.image_label.setStyleSheet("border: 2px solid black;")
        self.result_label = QLabel('Prediction: ', self)
        self.result_label.setStyleSheet("font-size: 16px;")


        layout = QVBoxLayout()
        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        layout.addWidget(self.upload_btn)
        layout.addWidget(self.image_label)
        layout.addWidget(self.predict_btn)
        layout.addWidget(self.result_label)
        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self.setLayout(layout)

        self.model = keras.models.load_model('mnist_cnn.h5')
        self.image_path = None

    def upload_image(self):
        file_dialog = QFileDialog()
        self.image_path, _ = file_dialog.getOpenFileName(self, 'Open Image File', '', 'Image files (*.jpg *.png *.bmp)')
        if self.image_path:
            pixmap = QPixmap(self.image_path)
            pixmap = pixmap.scaled(200, 200)
            self.image_label.setPixmap(pixmap)
            self.predict_btn.setEnabled(True)

    def preprocess_image(self, image_path):
        img = Image.open(image_path).convert('L')
        img = img.resize((28, 28), Image.LANCZOS)
        img = np.array(img) / 255.0
        img = img.reshape(1, 28, 28, 1)
        return img

    def predict_image(self):
        if self.image_path:
            img = self.preprocess_image(self.image_path)
            prediction = self.model.predict(img)
            predicted_class = np.argmax(prediction)
            self.result_label.setText(f'Prediction: {predicted_class}')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageClassifierApp()
    window.show()
    sys.exit(app.exec_())
