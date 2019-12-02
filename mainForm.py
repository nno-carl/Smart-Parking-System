# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'test.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.labelCamera = QtWidgets.QLabel(self.centralwidget)
        self.labelCamera.setGeometry(QtCore.QRect(220, 60, 300, 300))
        self.labelCamera.setObjectName("labelCamera")
        self.labelResult = QtWidgets.QLabel(self.centralwidget)
        self.labelResult.setGeometry(QtCore.QRect(340, 0, 130, 50))
        self.labelResult.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.labelResult.setObjectName("labelResult")
        self.btnReadImage_en = QtWidgets.QPushButton(self.centralwidget)
        self.btnReadImage_en.setGeometry(QtCore.QRect(170, 440, 90, 30))
        self.btnReadImage_en.setObjectName("btnReadImage_en")
        self.btnReadImage_ex = QtWidgets.QPushButton(self.centralwidget)
        self.btnReadImage_ex.setGeometry(QtCore.QRect(470, 440, 90, 30))
        self.btnReadImage_ex.setObjectName("btnReadImage_ex")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.btnReadImage_en.clicked.connect(MainWindow.btnReadImage_en_Clicked)
        self.btnReadImage_ex.clicked.connect(MainWindow.btnReadImage_ex_Clicked)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Parking"))
        self.labelCamera.setText(_translate("MainWindow", "Video"))
        self.labelResult.setText(_translate("MainWindow", "Parking Number"))
        self.btnReadImage_en.setText(_translate("MainWindow", "Entrance"))
        self.btnReadImage_ex.setText(_translate("MainWindow", "Exit"))

