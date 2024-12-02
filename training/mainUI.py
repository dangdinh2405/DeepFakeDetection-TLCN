import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, QComboBox
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
        self.setGeometry(100, 100, 800, 400)

        self.setWindowIcon(QIcon("training/icon/icon.png"))
        self.initUI()

        self.detector_path_capsule = 'E:/TLCN/Main/training/config/detector/capsule_net.yaml'
        self.weights_path_capsule = 'E:/TLCN/Main/training/weights/capsule_net_best.pth'

        self.detector_path_facexray = 'E:/TLCN/Main/training/config/detector/facexray.yaml'
        self.weights_path_facexray = 'E:/TLCN/Main/training/weights/facexray_best.pth'

        self.model = None
        
        self.setStyleSheet("background-color: #f4f4f4;")
        self.imagePath = None
        self.loadedImage = None

    def initUI(self):
        mainLayout = QHBoxLayout()
        
        self.imageLabel = QLabel("No Image")
        self.imageLabel.setFixedSize(300, 300)
        self.imageLabel.setStyleSheet("""
            background-color: gray;
            border: 2px solid black;
        """)
        self.imageLabel.setAlignment(Qt.AlignCenter)
        mainLayout.addWidget(self.imageLabel)

        buttonLayout = QVBoxLayout()

        buttonLayout.addStretch()
        buttonLayout.addStretch()
        buttonLayout.addStretch()
        buttonLayout.addStretch()
        buttonLayout.addStretch() 

        self.comboBox = QComboBox()
        self.comboBox.addItem("None")
        self.comboBox.addItem("Capsule Model")
        self.comboBox.addItem("FaceXRay Model")
        self.comboBox.setStyleSheet("""
            QComboBox {
                background-color: #f0f0f0;
                border: 2px solid #4a90e2;
                border-radius: 5px;
                padding: 5px;
                font-size: 16px;
            }
            QComboBox:hover {
                border-color: #0066cc;
            }
            QComboBox::editable {
                background: white;
            }
            QComboBox::drop-down {
                border: none;
                background: #f0f0f0;
            }
            QComboBox::down-arrow {
                image: url('arrow-icon.png');  /* Thêm biểu tượng cho mũi tên */
            }
            QComboBox QAbstractItemView {
                border: 1px solid #4a90e2;
                selection-background-color: #4a90e2;
                selection-color: white;
            }
        """)
        buttonLayout.addWidget(self.comboBox)

        # Kết nối sự kiện thay đổi lựa chọn của ComboBox với hàm tải mô hình
        self.comboBox.currentIndexChanged.connect(self.loadSelectedModel)

        buttonLayout.addStretch() 
  

        loadButton = QPushButton("Load Image")
        loadButton.setStyleSheet("""
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
        """)
        loadButton.clicked.connect(self.loadImage)
        
        buttonLayout.addWidget(loadButton)

        testButton = QPushButton("Test")
        testButton.setStyleSheet("""
            background-color: #008CBA;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
        """)
        testButton.clicked.connect(self.testImage)
        buttonLayout.addWidget(testButton)

        buttonLayout.addStretch()
        mainLayout.addLayout(buttonLayout)

        resultLayout = QVBoxLayout()
        resultLayout.addStretch()
        self.realConfidence = QLabel("Khả năng là ảnh thật: N/A")
        self.realConfidence.setStyleSheet("font-size: 16px; color: #333;")
        resultLayout.addWidget(self.realConfidence)

        self.fakeConfidence = QLabel("Khả năng là ảnh giả: N/A")
        self.fakeConfidence.setStyleSheet("font-size: 16px; color: #333;")
        self.fakeConfidence.setStyleSheet("font-size: 16px;")
        resultLayout.addWidget(self.fakeConfidence)

        self.inferenceTime = QLabel("Thời gian chạy: N/A")
        self.inferenceTime.setStyleSheet("font-size: 16px; color: red;")
        resultLayout.addWidget(self.inferenceTime)

        resultLayout.addStretch()
        resultLayout.addStretch()
        resultLayout.addStretch()

        self.creditLabel1 = QLabel("Thực hiện bởi:")
        self.creditLabel1.setStyleSheet("font-size: 14px; color: gray; font-weight: bold;")
        resultLayout.addWidget(self.creditLabel1)

        self.creditLabel2 = QLabel("21110837 - Nguyễn Quốc Lân")
        self.creditLabel2.setStyleSheet("font-size: 14px; color: gray;")
        resultLayout.addWidget(self.creditLabel2)

        self.creditLabel3 = QLabel("21110164 - Đinh Đại Hải Đăng")
        self.creditLabel3.setStyleSheet("font-size: 14px; color: gray;")
        resultLayout.addWidget(self.creditLabel3)

        self.creditLabel4 = QLabel("Giảng viên hướng dẫn:")
        self.creditLabel4.setStyleSheet("font-size: 14px; color: gray; font-weight: bold;")
        resultLayout.addWidget(self.creditLabel4)

        self.creditLabel5 = QLabel("Hoàng Văn Dũng")
        self.creditLabel5.setStyleSheet("font-size: 14px; color: gray;")
        resultLayout.addWidget(self.creditLabel5)

        resultLayout.addStretch()
        mainLayout.addLayout(resultLayout)

        centralWidget = QWidget()
        centralWidget.setLayout(mainLayout)
        self.setCentralWidget(centralWidget)

    def loadImage(self):
        filePath, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if filePath:
            self.imagePath = filePath
            self.loadedImage = cv2.imread(self.imagePath)
            self.loadedImage = cv2.cvtColor(self.loadedImage, cv2.COLOR_BGR2RGB)

            resizedImage = resizeToFit(self.loadedImage, 300, 300)
            finalImage = addPadding(resizedImage, 300, 300)

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
            self.realConfidence.setText(f"Khả năng là ảnh thật: {realProb:.2f}")
            self.fakeConfidence.setText(f"Khả năng là ảnh giả: {fakeProb:.2f}")
            self.inferenceTime.setText(f"Thời gian chạy: {inferenceTime:.2f} giây")  # Cập nhật nhãn credit2 để hiển thị thời gian
        else:
            self.realConfidence.setText("Khả năng là ảnh thật: N/A")
            self.fakeConfidence.setText("Khả năng là ảnh giả: N/A")
            self.inferenceTime.setText("Thời gian chạy: N/A")  # Cập nhật nếu không có ảnh

    def loadSelectedModel(self):
        # Lấy giá trị đã chọn từ ComboBox
        selected_model = self.comboBox.currentText()

        # Dựa trên lựa chọn, tải mô hình tương ứng
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
