# -*- coding: utf-8 -*-
import sys

from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt, QPoint
import numpy as np
from tensorflow import keras
from tensorflow.python.keras.models import *
G = load_model("G12500_1.2777_0.6862.h5")

def qimage2numpy(qimage, dtype='array'):
    """Convert QImage to numpy.ndarray.  The dtype defaults to uint8
    for QImage.Format_Indexed8 or `bgra_dtype` (i.e. a record array)
    for 32bit color images.  You can pass a different dtype to use, or
    'array' to get a 3D uint8 array for color images."""
    result_shape = (qimage.height(), qimage.width())
    temp_shape = (qimage.height(),
                  round(qimage.bytesPerLine() * 8 / qimage.depth()))
    if qimage.format() in (QImage.Format_ARGB32_Premultiplied,
                           QImage.Format_ARGB32,
                           QImage.Format_RGB32):
        if dtype == 'rec':
            dtype = QtGui.bgra_dtype
        elif dtype == 'array':
            dtype = np.uint8
            result_shape += (4,)
            temp_shape += (4,)
    elif qimage.format() == QImage.Format_Indexed8:
        dtype = np.uint8
    else:
        raise ValueError("qimage2numpy only supports 32bit and 8bit images")
        # FIXME: raise error if alignment does not match
    size = qimage.byteCount()
    buf = qimage.constBits()
    buf.setsize(size)
    result  = np.array(buf).reshape(temp_shape)
    # result = np.frombuffer(buf, dtype).reshape(temp_shape)
    if result_shape != temp_shape:
        result = result[:, :result_shape[1]]
    if qimage.format() == QImage.Format_RGB32 and dtype == np.uint8:
        result = result[..., :3]
    result = result[:,:,::-1]
    return result

def predict(data):
    w_noise = np.random.normal(0, 1, (1, 16, 16, 1024))
    source = data
    source = np.expand_dims(source,0)
    answer  = np.squeeze(G.predict([source,w_noise]),-1)
    answer =np.squeeze(answer,0)
    return answer
