# -*- coding: utf-8 -*-
"""
Demonstrate ability of ImageItem to be used as a canvas for painting with
the mouse.

"""



from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
app = pg.mkQApp("Draw Example")

## Create window with GraphicsView widget
w = pg.GraphicsView()
w.show()
w.resize(800,800)
w.setWindowTitle('pyqtgraph example: Draw')

view = pg.ViewBox()
w.setCentralItem(view)

## lock the aspect ratio
view.setAspectLocked(True)

## Create image item
img = pg.ImageItem(np.zeros((256,256)))
view.addItem(img)
view.setMouseEnabled(x=False,y=False)
view.setFixedHeight(256)
view.setFixedWidth(256)
## Set initial view bounds
# view.setRange(QtCore.QRectF(0, 0, 256, 256))

## start drawing with 3x3 brush
kern = np.ones([1,1],dtype=np.int32)
img.setDrawKernel(kern, mask=kern, center=(0,0), mode='set')
img.setLevels([0, 1])

if __name__ == '__main__':
    pg.exec()