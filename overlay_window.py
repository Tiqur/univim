from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPainter, QColor, QPen
from PyQt5.QtGui import QGuiApplication

class OverlayWindow(QMainWindow):
    update_signal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

        self.setWindowState(Qt.WindowFullScreen)
        self.setWindowFlags(self.windowFlags() | Qt.X11BypassWindowManagerHint)

        total_geometry = self.geometry()
        for screen in QGuiApplication.screens():
            total_geometry = total_geometry.united(screen.geometry())
        self.setGeometry(total_geometry)
    
        self.boxes = []
        self.update_signal.connect(self.update)

    def add_box(self, box):
        self.boxes.append(box)

    def clear_boxes(self):
        self.boxes = []

    def send_update_signal(self):
        self.update_signal.emit()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        painter.fillRect(self.rect(), Qt.transparent)

        # Draw cyan border
        pen = QPen(QColor(0, 255, 255))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawRect(self.rect().adjusted(2, 2, -2, -2))

        # Draw all boxes
        pen = QPen(QColor(0, 255, 255))
        painter.setPen(pen)
        for box in self.boxes:
            painter.drawRect(box)

        self.clear_boxes()

        # Show settings at top right
        max_settings_width = 105
        max_settings_height = 20
        painter.setBrush(QColor(0, 0, 0))
        painter.drawRect(0, 0, max_settings_width, max_settings_height)
        painter.drawText(5, 15, "Confidence: 0.5")
