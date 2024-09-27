from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtGui import QGuiApplication

class OverlayWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Set the window properties
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

        # Create a fullscreen window without a frame
        self.setWindowState(Qt.WindowFullScreen)
        self.setWindowFlags(self.windowFlags() | Qt.X11BypassWindowManagerHint)

        # Set geometry to cover all screens
        total_geometry = self.geometry()
        for screen in QGuiApplication.screens():
            total_geometry = total_geometry.united(screen.geometry())
        self.setGeometry(total_geometry)

    def paintEvent(self, event):
        # Handle the paint event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Set the background to transparent
        painter.fillRect(self.rect(), Qt.transparent)

        # Draw cyan border
        pen = QPen(QColor(0, 255, 255))  # Cyan color
        pen.setWidth(5)
        painter.setPen(pen)
        painter.drawRect(self.rect().adjusted(2, 2, -2, -2))  # Adjust to keep the border inside the window
