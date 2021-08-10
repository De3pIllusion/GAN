# -*- coding: utf-8 -*-
import sys

from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt, QPoint,QMetaObject,QCoreApplication
import numpy as np
from Global import qimage2numpy,predict
from matplotlib import pyplot as plt
import tensorflow as  tf

temp = np.zeros((256, 256))
temp2 = np.zeros((256, 256))
river = np.zeros((256, 256))
ridge = np.zeros((256, 256))
Data = [river,ridge,temp,temp2]
target = None
class Winform(QLabel):


    def __init__(self,feature):
        super(Winform, self).__init__(feature)
        # 设置标题
        self.setWindowTitle("绘图例子")
        self.feature = feature
        # 起点，终点
        self.lastPoint = QPoint()
        self.endPoint = QPoint()
        # 初始化
        self.initUi()
        self.setStyleSheet("border:1px solid black")
        self.setAlignment(Qt.AlignVCenter)


    def initUi(self):
        # 窗口大小设置为600*500

        self.resize(256, 256)
        # 画布大小为400*400，背景为白色
        self.pix = QPixmap(256, 256)
        self.pix.fill(Qt.white)
        self.setFixedSize(260, 260);



    def paintEvent(self, event):

        pp = QPainter(self.pix)
        # 根据鼠标指针前后两个位置绘制直线
        pp.drawLine(self.lastPoint, self.endPoint)
        # 让前一个坐标值等于后一个坐标值，
        # 这样就能实现画出连续的线
        self.lastPoint = self.endPoint
        painter = QPainter(self)
        # 绘制画布到窗口指定位置处
        painter.drawPixmap(2,2, self.pix)


    def mousePressEvent(self, event):
        # 鼠标左键按下
        if event.button() == Qt.LeftButton:
            self.lastPoint = event.pos()
            self.endPoint = self.lastPoint


    def mouseMoveEvent(self, event):
        # 鼠标左键按下的同时移动鼠标
        if event.buttons() and Qt.LeftButton:
            self.endPoint = event.pos()
            # 进行重新绘制
            self.update()


    def mouseReleaseEvent(self, event):
        # 鼠标左键释放
        if event.button() == Qt.LeftButton:
            self.endPoint = event.pos()
        # 进行重新绘制
        self.myupdate()
        self.update()

    def myupdate(self):
        global Data,target
        self.myarray = qimage2numpy(self.pix.toImage())
        self.source = np.ones((256,256))*255
        for i in range(256):
            for j in range(256):
                self.source[i][j] = self.myarray[i][j][0]

        if(self.feature == "ridge"):
            Data[1] = self.source
        else:
            Data[0] = self.source
        temp1 = Data[0]
        temp2 = Data[1]
        temp3 = Data[2]
        temp4 = Data[3]
        target =np.squeeze(
                np.stack(
                    (np.expand_dims(temp1, -1),
                     np.expand_dims(temp2, -1),
                     np.expand_dims(temp3, -1),
                     np.expand_dims(temp4, -1)),
                    axis=2
                ),
                axis=-1
            )
        w_noise = np.random.normal(0, 1, (1, 16, 16, 1024))
        for i in range(4):
            for j in range(target.shape[0]):
                for m in range(target.shape[1]):
                    if (target[j][m][i] > 0): target[j][m][i] = 1.0

        pred =predict(target)
        answer = pred*1000+127
        print(answer)
        plt.imshow(np.squeeze(pred)*1000+127)
        plt.show()
        print(pred.shape)

if __name__ =="__main__":



    app = QApplication(sys.argv)
    form = Winform("ridge")
    form2 =Winform("river")

    window = QMainWindow()
    window.resize(768,1000)

    Hlayout =QHBoxLayout(window)
    Hlayout.addWidget(form)
    Hlayout.addWidget(form2)


    myWidget = QWidget()
    myWidget.setLayout(Hlayout)


    window.setCentralWidget(myWidget)
    window.show()
    sys.exit(app.exec_())
