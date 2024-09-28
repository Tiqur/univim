from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtCore import Qt, pyqtSignal, QRect, QThreadPool, QRunnable
from PyQt5.QtGui import QGuiApplication, QPainter, QColor, QPen, QFont, QFontMetrics, QScreen
import sys
import threading

class YOLOLoader(QRunnable):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def run(self):
        from ultralytics import YOLO
        model = YOLO("weights/best.pt")
        self.callback(model)

class OverlayWindow(QMainWindow):
    update_signal = pyqtSignal()
    model_loaded_signal = pyqtSignal()

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
        self.model_loaded_signal.connect(self.on_model_loaded)
        self.sequence = 1
        self.font = QFont("Arial", 12)
        self.is_rendering = False
        self.model = None
        self.thread_pool = QThreadPool()

        # Preload the YOLO model
        self.preload_model()

    def preload_model(self):
        loader = YOLOLoader(self.set_model)
        self.thread_pool.start(loader)

    def set_model(self, model):
        self.model = model
        self.model_loaded_signal.emit()

    def on_model_loaded(self):
        print("YOLO model loaded successfully")

    def start(self):
        print("Overlay window loaded successfully")
        self.show()

    def screenshot(self):
        screen = QApplication.primaryScreen()
        screenshot = screen.grabWindow(0)
        screenshot.save('screenshot.png', 'png')

    def render_start(self):
        if self.model is None:
            print("YOLO model is not loaded yet. Please wait.")
            return

        self.screenshot()
        self.is_rendering = True
        threading.Thread(target=self.run_yolo).start()

    def render_stop(self):
        print("STOPPING RENDERING")
        self.is_rendering = False
        self.boxes = []
        self.update()

    def run_yolo(self):
        results = self.model("screenshot.png", conf=0.1, max_det=2048, iou=0.4, stream=True)
        
        for result in results:
            all_boxes = []
            for box in result.boxes:
                dim = (box.xywhn).tolist()[0]
                orig_width = box.orig_shape[1]
                orig_height = box.orig_shape[0]

                width = int(dim[2]*orig_width)
                height = int(dim[3]*orig_height)
                x = int(dim[0]*orig_width-width/2)
                y = int(dim[1]*orig_height-height/2)

                new_box = QRect(x, y, width, height)
                all_boxes.append(new_box)

            filtered_boxes = self.filter_boxes(all_boxes)
            
            for box in filtered_boxes:
                self.add_box(box)

            self.send_update_signal()


    def secondary_thread(self):
        results = self.model("screenshot.png", conf=0.1, max_det=2048, iou=0.4, stream=True)
        
        for result in results:
            all_boxes = []
            for box in result.boxes:
                dim = (box.xywhn).tolist()[0]
                orig_width = box.orig_shape[1]
                orig_height = box.orig_shape[0]

                width = int(dim[2]*orig_width)
                height = int(dim[3]*orig_height)
                x = int(dim[0]*orig_width-width/2)
                y = int(dim[1]*orig_height-height/2)

                new_box = QRect(x, y, width, height)
                all_boxes.append(new_box)

            filtered_boxes = self.filter_boxes(all_boxes)
            
            for box in filtered_boxes:
                self.add_box(box)

            self.send_update_signal()


    def is_significant_overlap(self, box1, box2, threshold=0.2):
        intersect = box1.intersected(box2)
        intersect_area = intersect.width() * intersect.height()
        smaller_box_area = min(box1.width() * box1.height(), box2.width() * box2.height())
        return intersect_area / smaller_box_area > threshold

    def filter_boxes(self, boxes):
        sorted_boxes = sorted(boxes, key=lambda b: b.width() * b.height())
        
        filtered_boxes = []
        for i, box in enumerate(sorted_boxes):
            should_add = True
            for j in range(i):
                if self.is_significant_overlap(box, sorted_boxes[j]):
                    should_add = False
                    break
            if should_add:
                filtered_boxes.append(box)
        
        return filtered_boxes

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

    def draw_box_label(self, painter, p):
        x = p.x()
        y = p.y()

        # Set the font and get the metrics
        painter.setFont(self.font)
        metrics = QFontMetrics(self.font)

        # Get the width and height of the text
        shortcut = self.get_next_letter()
        text_width_px = metrics.width(shortcut)
        font_height_px = metrics.height()

        # Define padding relative to the font size
        padding_x = int(font_height_px * 0.1)  # Horizontal padding
        padding_y = int(font_height_px * 0.1)  # Vertical padding

        # Define the rectangle size based on the text size and padding
        rect_width = int(text_width_px + 2 * padding_x)
        rect_height = int(font_height_px + 2 * padding_y)

        # Set the pen color and fill the rectangle behind the text
        painter.fillRect(x, y, rect_width, rect_height, QColor(0, 255, 255))
        painter.setPen(QColor(0, 0, 0))

        # Calculate centered positions for the text
        text_x = x + (rect_width - text_width_px) // 2
        text_y = y + (rect_height + metrics.ascent() - metrics.descent()) // 2

        # Draw the text at the calculated position
        painter.drawText(text_x, text_y, shortcut)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        painter.fillRect(self.rect(), Qt.transparent)

        if not self.is_rendering:
            return

        # Draw cyan border
        pen = QPen(QColor(0, 255, 255))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawRect(self.rect().adjusted(2, 2, -2, -2))

        # Draw all boxes
        pen = QPen(QColor(0, 255, 255))
        painter.setPen(pen)
        for box in self.boxes:
            tl_point = box.topLeft()
            self.draw_box_label(painter, tl_point)
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
