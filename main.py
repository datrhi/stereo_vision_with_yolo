
import numpy as np
import cv2
import time
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from threading import Thread
from worker import Worker1
start = True
dataset = None
cfg = None
weight = None
video_source_index = None
# Constant
base_line = 120
focal_length = 3
mapMMToPixel = 375
listPort = []
constant = base_line * focal_length * mapMMToPixel * 0.48
FRAME_WIDTH = 1280
FRAME_HEIGHT = 480


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1450, 872)
        MainWindow.setMaximumSize(QtCore.QSize(1920, 1080))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setMaximumSize(QtCore.QSize(1920, 1080))
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(520, 660, 371, 151))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(920, 660, 91, 41))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(920, 770, 91, 41))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_9 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_9.setGeometry(QtCore.QRect(80, 660, 141, 51))
        self.pushButton_9.setObjectName("pushButton_9")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(230, 660, 261, 51))
        self.label.setText("")
        self.label.setObjectName("label")
        self.pushButton_10 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_10.setGeometry(QtCore.QRect(80, 710, 141, 51))
        self.pushButton_10.setObjectName("pushButton_10")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(230, 710, 261, 51))
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.pushButton_11 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_11.setGeometry(QtCore.QRect(80, 760, 141, 51))
        self.pushButton_11.setObjectName("pushButton_11")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(230, 760, 261, 51))
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(1040, 730, 231, 81))
        self.pushButton_3.setObjectName("pushButton_3")
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(1040, 690, 231, 31))
        self.comboBox.setObjectName("comboBox")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(1040, 660, 201, 21))
        self.label_4.setStyleSheet("font: 12pt \"MS Shell Dlg 2\";")
        self.label_4.setObjectName("label_4")
        self.FeedLabel = QtWidgets.QLabel(self.centralwidget)
        self.FeedLabel.setGeometry(QtCore.QRect(100, 20, 1280, 480))
        # self.FeedLabel.setText("")
        self.FeedLabel.setObjectName("FeedLabel")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1450, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        global listPort
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate(
            "MainWindow", "Depth Estimation"))
        self.pushButton.setText(_translate("MainWindow", "Start"))
        self.pushButton_2.setText(_translate("MainWindow", "Stable mode"))
        self.pushButton_4.setText(_translate("MainWindow", "Debug mode"))
        self.pushButton_9.setText(_translate(
            "MainWindow", "Choose dataset file"))
        self.pushButton_10.setText(
            _translate("MainWindow", "Choose cfg file"))
        self.pushButton_11.setText(_translate(
            "MainWindow", "Choose weight file"))
        self.pushButton_3.setText(_translate("MainWindow", "Calibrate"))
        self.label_4.setText(_translate(
            "MainWindow", "Video source index"))
        self.pushButton_9.clicked.connect(self.uploadDataset)
        self.pushButton_10.clicked.connect(self.uploadCfg)
        self.pushButton_11.clicked.connect(self.uploadWeight)
        self.pushButton.clicked.connect(self.btn_thread)
        _, listPort = self.list_ports()
        self.comboBox.addItems([str(port) for port in listPort])
        self.comboBox.currentTextChanged.connect(
            self.video_source_on_change)

    def ImageUpdateSlot(self, Image):
        self.FeedLabel.setPixmap(QPixmap.fromImage(Image))

    def list_ports(self):
        is_working = True
        dev_port = 0
        working_ports = []
        available_ports = []
        while is_working:
            camera = cv2.VideoCapture(dev_port)
            if not camera.isOpened():
                is_working = False
                # print("Port %s is not working." % dev_port)
            else:
                is_reading, img = camera.read()
                w = camera.get(3)
                h = camera.get(4)
                if is_reading:
                    # print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
                    working_ports.append(dev_port)
                else:
                    # print("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
                    available_ports.append(dev_port)
            dev_port += 1
        return available_ports, working_ports

    def video_source_on_change(self, current):
        global video_source_index
        if current is not None:
            video_source_index = int(current)

    def btn_thread(self):
        global start, dataset, cfg, weight, video_source_index, listPort
        if dataset is not None and cfg is not None and weight is not None and len(listPort) > 0:
            if video_source_index is None:
                video_source_index = listPort[0]
            if start:
                self.pushButton.setText("Stop")
                start = False
                self.Worker1 = Worker1(
                    video_source_index, dataset, cfg, weight)

                self.Worker1.start()
                self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)
            else:
                self.pushButton.setText("Start")
                start = True
                self.Worker1.stop()

        else:
            msg = QtWidgets.QMessageBox()
            msg.setWindowTitle("Error")
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("Please set up config first!!")
            msg.exec_()

    def uploadDataset(self):
        global dataset
        dataset, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "Open File", "", "All Files (*)")
        if dataset:
            self.label.setText(dataset)

    def uploadCfg(self):
        global cfg
        cfg, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "Open File", "", "All Files (*)")
        if cfg:
            self.label_2.setText(cfg)

    def uploadWeight(self):
        global weight
        weight, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "Open File", "", "All Files (*)")
        if weight:
            self.label_3.setText(weight)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
