# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'task4.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(470, 431)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.table1 = QtWidgets.QTableWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.table1.sizePolicy().hasHeightForWidth())
        self.table1.setSizePolicy(sizePolicy)
        self.table1.setBaseSize(QtCore.QSize(0, 0))
        self.table1.setLineWidth(1)
        self.table1.setGridStyle(QtCore.Qt.SolidLine)
        self.table1.setRowCount(7)
        self.table1.setColumnCount(1)
        self.table1.setObjectName("table1")
        self.table1.horizontalHeader().setVisible(False)
        self.table1.horizontalHeader().setCascadingSectionResizes(False)
        self.table1.verticalHeader().setDefaultSectionSize(41)
        self.gridLayout.addWidget(self.table1, 2, 0, 1, 2)
        spacerItem = QtWidgets.QSpacerItem(89, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 1, 2, 1, 1)
        self.Browse1 = QtWidgets.QPushButton(self.centralwidget)
        self.Browse1.setObjectName("Browse1")
        self.gridLayout.addWidget(self.Browse1, 0, 0, 1, 1)
        self.mixer = QtWidgets.QSlider(self.centralwidget)
        self.mixer.setOrientation(QtCore.Qt.Horizontal)
        self.mixer.setObjectName("mixer")
        self.gridLayout.addWidget(self.mixer, 1, 0, 1, 2)
        self.Browse2 = QtWidgets.QPushButton(self.centralwidget)
        self.Browse2.setObjectName("Browse2")
        self.gridLayout.addWidget(self.Browse2, 0, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.table1.setSortingEnabled(False)
        self.Browse1.setText(_translate("MainWindow", "Browse1"))
        self.Browse2.setText(_translate("MainWindow", "Browse2"))

