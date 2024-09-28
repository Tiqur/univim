from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtCore import Qt, pyqtSignal, QRect, QThreadPool, QRunnable, QTimer
from PyQt5.QtGui import QGuiApplication, QPainter, QColor, QPen, QFont, QFontMetrics, QScreen
import sys
import threading

class YOLOModelLoader(QRunnable):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def run(self):
        from ultralytics import YOLO
        model = YOLO("weights/best.pt")
        self.callback(model)

class OverlayWindow(QMainWindow):
    update_overlay_signal = pyqtSignal()
    model_loaded_signal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_window_properties()
        self.initialize_variables()
        self.setup_signals()
        self.preload_ai_model()

    def setup_window_properties(self):
        # Set up the window to be frameless, stay on top, and cover the full screen
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.X11BypassWindowManagerHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setWindowState(Qt.WindowFullScreen)
        self.set_fullscreen_geometry()

    def set_fullscreen_geometry(self):
        # Ensure the window covers all available screens
        total_geometry = self.geometry()
        for screen in QGuiApplication.screens():
            total_geometry = total_geometry.united(screen.geometry())
        self.setGeometry(total_geometry)

    def initialize_variables(self):
        self.clickable_elements = []
        self.sequence_counter = 1
        self.label_font = QFont("Arial", 12)
        self.is_overlay_active = False
        self.ai_model = None
        self.thread_pool = QThreadPool()

    def setup_signals(self):
        self.update_overlay_signal.connect(self.update)
        self.model_loaded_signal.connect(self.on_ai_model_loaded)

    def preload_ai_model(self):
        # Start loading the AI model in a separate thread
        loader = YOLOModelLoader(self.set_ai_model)
        self.thread_pool.start(loader)

    def set_ai_model(self, model):
        self.ai_model = model
        self.model_loaded_signal.emit()

    def on_ai_model_loaded(self):
        print("AI model loaded successfully")

    def activate_overlay(self):
        print("Overlay window running")
        self.show()

    def capture_screen(self):
        # Capture the entire screen
        screen = QApplication.primaryScreen()
        screenshot = screen.grabWindow(0)
        screenshot.save('current_screen.png', 'png')

    def start_element_detection(self):
        if self.ai_model is None:
            print("AI model is not loaded yet. Please wait.")
            QTimer.singleShot(1000, self.start_element_detection)  # Retry after 1 second
            return

        self.capture_screen()
        self.is_overlay_active = True
        threading.Thread(target=self.detect_clickable_elements).start()

    def stop_element_detection(self):
        print("Stopping element detection")
        self.is_overlay_active = False
        self.clickable_elements = []
        self.update()

    def detect_clickable_elements(self):
        results = self.ai_model("current_screen.png", conf=0.1, max_det=2048, iou=0.4, stream=True)
        
        for result in results:
            all_elements = self.extract_elements_from_result(result)
            filtered_elements = self.filter_overlapping_elements(all_elements)
            
            self.clickable_elements = filtered_elements
            self.update_overlay_signal.emit()

    def extract_elements_from_result(self, result):
        elements = []
        for box in result.boxes:
            dim = (box.xywhn).tolist()[0]
            orig_width, orig_height = box.orig_shape[1], box.orig_shape[0]
            width = int(dim[2] * orig_width)
            height = int(dim[3] * orig_height)
            x = int(dim[0] * orig_width - width/2)
            y = int(dim[1] * orig_height - height/2)
            elements.append(QRect(x, y, width, height))
        return elements

    def filter_overlapping_elements(self, elements, overlap_threshold=0.2):
        sorted_elements = sorted(elements, key=lambda e: e.width() * e.height())
        filtered_elements = []
        for i, element in enumerate(sorted_elements):
            if all(not self.is_significant_overlap(element, filtered_elem, overlap_threshold) 
                   for filtered_elem in filtered_elements):
                filtered_elements.append(element)
        return filtered_elements

    def is_significant_overlap(self, elem1, elem2, threshold):
        intersect = elem1.intersected(elem2)
        intersect_area = intersect.width() * intersect.height()
        smaller_elem_area = min(elem1.width() * elem1.height(), elem2.width() * elem2.height())
        return intersect_area / smaller_elem_area > threshold

    def to_base_26(self, num):
        # Convert a number to a base-26 representation using letters
        result = []
        while num > 0:
            num -= 1
            result.append(chr(num % 26 + ord('a')))
            num //= 26
        return ''.join(reversed(result))

    def get_next_label(self):
        label = self.to_base_26(self.sequence_counter)
        self.sequence_counter += 1
        return label

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), Qt.transparent)

        if not self.is_overlay_active:
            return

        self.draw_overlay_border(painter)
        self.draw_clickable_elements(painter)
        self.draw_settings_info(painter)

        self.clickable_elements = []
        self.sequence_counter = 1

    def draw_overlay_border(self, painter):
        # Draw a cyan border around the entire screen
        pen = QPen(QColor(0, 255, 255))  # Cyan color
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawRect(self.rect().adjusted(2, 2, -2, -2))

    def draw_clickable_elements(self, painter):
        painter.setPen(QPen(QColor(0, 255, 255)))  # Cyan color for element borders
        for element in self.clickable_elements:
            self.draw_element_label(painter, element.topLeft())
            painter.drawRect(element)

    def draw_element_label(self, painter, position):
        x, y = position.x(), position.y()
        painter.setFont(self.label_font)
        metrics = QFontMetrics(self.label_font)

        label = self.get_next_label()
        text_width = metrics.width(label)
        font_height = metrics.height()

        padding_x = int(font_height * 0.1)
        padding_y = int(font_height * 0.1)

        rect_width = text_width + 2 * padding_x
        rect_height = font_height + 2 * padding_y

        painter.fillRect(x, y, rect_width, rect_height, QColor(0, 255, 255))  # Cyan background
        painter.setPen(QColor(0, 0, 0))  # Black text

        text_x = x + (rect_width - text_width) // 2
        text_y = y + (rect_height + metrics.ascent() - metrics.descent()) // 2

        painter.drawText(text_x, text_y, label)

    def draw_settings_info(self, painter):
        # Draw settings information in the top-left corner
        settings_width, settings_height = 105, 35
        painter.fillRect(0, 0, settings_width, settings_height, QColor(0, 0, 0))
        painter.setPen(QColor(255, 255, 255))  # White text
        painter.drawText(5, 15, "Confidence: 0.5")
        painter.drawText(5, 30, "Overlap: 0.7")
