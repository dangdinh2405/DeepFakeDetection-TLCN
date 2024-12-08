import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, QComboBox, QGridLayout
)
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt
import cv2
import numpy as np
from utilUI import infer, load_model
import time

def resizeToFit(image, targetWidth, targetHeight):
    originalHeight, originalWidth = image.shape[:2]
    ratio = min(targetWidth / originalWidth, targetHeight / originalHeight)
    newWidth = int(originalWidth * ratio)
    newHeight = int(originalHeight * ratio)
    resizedImage = cv2.resize(image, (newWidth, newHeight), interpolation=cv2.INTER_AREA)
    return resizedImage


def addPadding(image, targetWidth, targetHeight):
    height, width = image.shape[:2]
    deltaW = targetWidth - width
    deltaH = targetHeight - height
    top, bottom = deltaH // 2, deltaH - (deltaH // 2)
    left, right = deltaW // 2, deltaW - (deltaW // 2)
    color = [128, 128, 128]  
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


class ImageClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ứng dụng dự đoán ảnh deepfake")
        self.setGeometry(100, 100, 800, 600)

        self.setWindowIcon(QIcon("training/icon/icon.png"))
        self.initUI()

        self.detector_path_capsule = 'training/config/detector/capsule_net.yaml'
        self.weights_path_capsule = 'training/weights/capsule_net_best.pth'

        self.detector_path_facexray = 'training/config/detector/facexray.yaml'
        self.weights_path_facexray = 'training/weights/facexray_best.pth'

        self.model = None
        
        self.setStyleSheet("background-color: #f4f4f4;")
        self.imagePath = None
        self.loadedImage = None

    def initUI(self):
        # Main layout
        mainLayout = QGridLayout()

        # Left panel: Results and credits
        resultLayout = QVBoxLayout()
        self.realConfidence = QLabel("Khả năng là ảnh thật: N/A")
        self.realConfidence.setStyleSheet("font-size: 16px; color: #4a90e2; font-weight: bold;")
        resultLayout.addWidget(self.realConfidence)

        self.fakeConfidence = QLabel("Khả năng là ảnh giả: N/A")
        self.fakeConfidence.setStyleSheet("font-size: 16px; color: #e74c3c; font-weight: bold;")
        resultLayout.addWidget(self.fakeConfidence)

        self.inferenceTime = QLabel("Thời gian chạy: N/A")
        self.inferenceTime.setStyleSheet("font-size: 16px; color: #2ecc71; font-weight: bold;")
        resultLayout.addWidget(self.inferenceTime)

        resultLayout.addStretch()

        creditTitle = QLabel("Thực hiện bởi:")
        creditTitle.setStyleSheet("font-size: 14px; color: gray; font-weight: bold;")
        resultLayout.addWidget(creditTitle)

        credits = [
            "21110837 - Nguyễn Quốc Lân",
            "21110164 - Đinh Đại Hải Đăng",
            "Giảng viên: Hoàng Văn Dũng"
        ]
        for credit in credits:
            label = QLabel(credit)
            label.setStyleSheet("font-size: 14px; color: gray;")
            resultLayout.addWidget(label)

        mainLayout.addLayout(resultLayout, 0, 0, 2, 1)

        # Center: Image display
        self.imageLabel = QLabel("No Image")
        self.imageLabel.setFixedSize(400, 400)
        self.imageLabel.setStyleSheet("""
            background-color: #dcdde1;
            border: 2px solid #7f8c8d;
            font-size: 18px;
            color: #2c3e50;
        """)
        self.imageLabel.setAlignment(Qt.AlignCenter)
        mainLayout.addWidget(self.imageLabel, 0, 1, 2, 1)

        # Right panel: Buttons
        buttonLayout = QVBoxLayout()
        self.comboBox = QComboBox()
        self.comboBox.addItems(["None", "Capsule Model", "FaceXRay Model"])
        self.comboBox.setStyleSheet("""
            QComboBox {
                background-color: white;
                border: 1px solid #95a5a6;
                border-radius: 5px;
                padding: 5px;
                font-size: 14px;
            }
            QComboBox:hover {
                border-color: #3498db;
            }
        """)
        self.comboBox.currentIndexChanged.connect(self.loadSelectedModel)
        buttonLayout.addWidget(self.comboBox)

        loadButton = QPushButton("Load Image")
        loadButton.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        loadButton.clicked.connect(self.loadImage)
        buttonLayout.addWidget(loadButton)

        testButton = QPushButton("Test")
        testButton.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 10px;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        testButton.clicked.connect(self.testImage)
        buttonLayout.addWidget(testButton)

        buttonLayout.addStretch()

        instructions = QLabel("""
        Hướng dẫn sử dụng:
        1. Chọn mô hình phát hiện từ danh sách.
        2. Tải ảnh cần phân tích bằng nút 'Load Image'.
        3. Nhấn 'Test' để bắt đầu phân tích.
        """)
        instructions.setStyleSheet("""
            font-size: 14px; 
            color: #7f8c8d;  
            padding: 10px; 
            border-radius: 5px;
        """)
        buttonLayout.addWidget(instructions)
        mainLayout.addLayout(buttonLayout, 0, 2, 2, 1)

        # Set main layout
        centralWidget = QWidget()
        centralWidget.setLayout(mainLayout)
        self.setCentralWidget(centralWidget)


    def loadImage(self):
        filePath, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if filePath:
            self.imagePath = filePath
            self.loadedImage = cv2.imread(self.imagePath)
            self.loadedImage = cv2.cvtColor(self.loadedImage, cv2.COLOR_BGR2RGB)

            resizedImage = resizeToFit(self.loadedImage, 400, 400)
            finalImage = addPadding(resizedImage, 400, 400)

            height, width, channel = finalImage.shape
            bytesPerLine = channel * width
            qImage = QImage(finalImage.data, width, height, bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qImage)

            self.imageLabel.setPixmap(pixmap)
    def testImage(self):
        if self.model is None:
            print("Vui lòng chọn và tải mô hình trước khi tải ảnh.")
            return

        if self.imagePath:
            startTime = time.time()  # Bắt đầu đo thời gian
            realProb, fakeProb = infer(self.model, self.imagePath)
            endTime = time.time()  # Kết thúc đo thời gian

            inferenceTime = endTime - startTime  # Tính thời gian inference

            # Hiển thị kết quả
            self.realConfidence.setText(f"Khả năng là ảnh thật: {realProb:.4f}")
            self.fakeConfidence.setText(f"Khả năng là ảnh giả: {fakeProb:.4f}")
            self.inferenceTime.setText(f"Thời gian chạy: {inferenceTime:.2f} giây") 
        else:
            self.realConfidence.setText("Khả năng là ảnh thật: N/A")
            self.fakeConfidence.setText("Khả năng là ảnh giả: N/A")
            self.inferenceTime.setText("Thời gian chạy: N/A")

    def loadSelectedModel(self):
        selected_model = self.comboBox.currentText()

        if selected_model == "Capsule Model":
            self.model = load_model(self.weights_path_capsule, self.detector_path_capsule)
            print("Đã tải mô hình Capsule.")
        elif selected_model == "FaceXRay Model":
            self.model = load_model(self.weights_path_facexray, self.detector_path_facexray)
            print("Đã tải mô hình FaceXRay.")
        else:
            print("Không có mô hình nào được chọn.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = ImageClassifierApp()
    mainWindow.show()
    sys.exit(app.exec_())
