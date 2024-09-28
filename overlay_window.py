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

        self.setStyleSheet('font-size: 16px')

        self.setWindowState(Qt.WindowFullScreen)
        self.setWindowFlags(self.windowFlags() | Qt.X11BypassWindowManagerHint)

        total_geometry = self.geometry()
        for screen in QGuiApplication.screens():
            total_geometry = total_geometry.united(screen.geometry())
        self.setGeometry(total_geometry)
    
        self.boxes = []
        self.update_signal.connect(self.update)
        self.sequence = 1


    def to_base_26(self, num):
        """Convert a number to a base-26 representation using letters."""
        result = []
        while num > 0:
            num -= 1  # Adjust for 0-indexing
            result.append(chr(num % 26 + ord('a')))
            num //= 26
        return ''.join(reversed(result))

    def get_next_letter(self):
        """Return the next letter in the sequence and increment the counter."""
        letter = self.to_base_26(self.sequence)  # Get the current letter
        self.sequence += 1  # Increment to prepare for the next call
        return letter


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
        # In the future, draw all at once using painter.drawRects(Rect[])
        pen = QPen(QColor(0, 255, 255))
        painter.setPen(pen)
        for box in self.boxes:
            tl_point = box.topLeft()
            x = tl_point.x()
            y = tl_point.y()
            pen = QPen(QColor(0, 0, 0))
            painter.fillRect(x, y-15, 30, 15, QColor(0, 255, 255))
            painter.setPen(QColor(0, 0, 0))
            shortcut = self.get_next_letter()
            painter.drawText(x, y-2, shortcut)
            painter.setPen(QColor(0, 255, 255))
            painter.drawRect(box)

        self.clear_boxes()
        self.sequence = 1

        # Show settings at top right
        max_settings_width = 105
        max_settings_height = 35
        painter.setBrush(QColor(0, 0, 0))
        painter.drawRect(0, 0, max_settings_width, max_settings_height)
        painter.drawText(5, 15, "Confidence: 0.5")
        painter.drawText(5, 30, "Overlap: 0.7")
