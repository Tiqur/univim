from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtCore import Qt, pyqtSignal, QRect, QThreadPool, QRunnable, QTimer, QPoint
from PyQt5.QtGui import QGuiApplication, QPainter, QColor, QPen, QFont, QFontMetrics, QScreen, QBrush
import sys
import threading
from pynput.mouse import Controller as MouseController, Button
import itertools

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
    key_pressed_signal = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_window_properties()
        self.initialize_variables()
        self.setup_signals()
        self.preload_ai_model()
        self.mouse = MouseController()
        self.current_input = ''
        self.is_grid_view_active = False

    def setup_window_properties(self):
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setWindowState(Qt.WindowFullScreen)
        self.setWindowFlags(self.windowFlags() | Qt.X11BypassWindowManagerHint)
        self.set_fullscreen_geometry()

    def set_fullscreen_geometry(self):
        total_geometry = self.geometry()
        for screen in QGuiApplication.screens():
            total_geometry = total_geometry.united(screen.geometry())
        self.setGeometry(total_geometry)

    def initialize_variables(self):
        self.clickable_elements = []
        self.element_labels = []
        self.label_font = QFont("Arial", 12)
        self.is_overlay_active = False
        self.ai_model = None
        self.thread_pool = QThreadPool()
        self.is_grid_view_active = False

    def setup_signals(self):
        self.update_overlay_signal.connect(self.update)
        self.model_loaded_signal.connect(self.on_ai_model_loaded)
        self.key_pressed_signal.connect(self.handle_key_press)

    def preload_ai_model(self):
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
        screen = QApplication.primaryScreen()
        screenshot = screen.grabWindow(0)
        screenshot.save('current_screen.png', 'png')

    def start_element_detection(self):
        self.current_input = ''
        if self.ai_model is None:
            print("AI model is not loaded yet. Please wait.")
            QTimer.singleShot(1000, self.start_element_detection)
            return

        self.capture_screen()
        self.is_overlay_active = True
        threading.Thread(target=self.detect_clickable_elements).start()

    def stop_element_detection(self):
        print("Stopping element detection")
        self.is_overlay_active = False
        self.clickable_elements = []
        self.element_labels = []
        self.update()

    def detect_clickable_elements(self):
        results = self.ai_model("current_screen.png", conf=0.1, max_det=2048, iou=0.4, stream=True)
        
        for result in results:
            all_elements = self.extract_elements_from_result(result)
            filtered_elements = self.filter_overlapping_elements(all_elements)
            
            self.clickable_elements = filtered_elements
            self.element_labels = self.generate_labels(len(filtered_elements))
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

    def generate_labels(self, count):
        labels = []
        characters = 'abcdefghijklmnopqrstuvwxyz'
        
        # Step 1: Reserve the last x letters for single-character labels (default to 12)
        single_char_limit = min(count, 12)
        single_char_labels = list(characters[single_char_limit:])
        labels.extend(single_char_labels)
        
        # Step 2: If more labels are needed, generate multi-character labels dynamically
        remaining_count = count - len(labels)
        
        if remaining_count > 0:
            current_length = 2  # Start with two-character labels
            
            while remaining_count > 0:
                first_characters = characters[:single_char_limit]  # Use reserved first characters
                # Dynamically generate labels of current length
                for combo in itertools.product(first_characters, repeat=current_length):
                    labels.append(''.join(combo))
                    if len(labels) == count:  # Stop once we reach the desired count
                        return labels
                
                # Increase the length for the next iteration
                current_length += 1
                remaining_count = count - len(labels)  # Update the remaining count

        return labels[:count]

    def handle_key_press(self, key):
        if self.is_overlay_active:
            self.current_input += key
            self.update_labels_starting_with(self.current_input)
            self.update()

    def toggle_grid_view(self):
        self.is_grid_view_active = not self.is_grid_view_active
        self.is_overlay_active = False
        self.update()

    def stop_grid_view(self):
        self.is_grid_view_active = False
        self.update()


    def update_labels_starting_with(self, input_string):
        matching_labels = [label for label in self.element_labels if label.startswith(input_string)]
        
        if not matching_labels:
            # Reset if there are no matches
            self.current_input = ''
            return

        if len(matching_labels) == 1:
            # Exact match found
            matched_label = matching_labels[0]
            if input_string == matched_label:
                index = self.element_labels.index(matched_label)
                self.click_element(index)
                self.clickable_elements.pop(index)
                self.element_labels.pop(index)
                self.current_input = ''  # Reset input after clicking
            else:
                # Update visible label for the unique match
                index = self.element_labels.index(matched_label)
                self.element_labels[index] = matched_label
        else:
            # Update visible labels for partial matches
            for i, label in enumerate(self.element_labels):
                if label.startswith(input_string):
                    self.element_labels[i] = label
                else:
                    self.element_labels[i] = ''
        # Remove elements with empty labels
        self.clickable_elements = [elem for elem, label in zip(self.clickable_elements, self.element_labels) if label]
        self.element_labels = [label for label in self.element_labels if label]

        self.update()

    def scroll_up(self):
        self.mouse.scroll(0, 2)

    def scroll_down(self):
        self.mouse.scroll(0, -2)

    def click_element(self, index):
        element = self.clickable_elements[index]
        center = element.center()
        self.mouse.position = (center.x(), center.y())
        self.mouse.click(Button.left)
        print(f"Clicked element at {center.x()}, {center.y()}")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), Qt.transparent)

        if not self.is_overlay_active and not self.is_grid_view_active:
            return

        self.draw_overlay_border(painter)
        
        if self.is_overlay_active:
            self.draw_clickable_elements(painter)
            self.draw_settings_info(painter)
        
        if self.is_grid_view_active:
            self.draw_grid(painter)

    def draw_grid(self, painter):
        pen = QPen(QColor(0, 255, 255))
        pen.setWidth(2)
        painter.setPen(pen)

        div = 26  # Cell divisions to make

        # Calculate the width and height of each grid cell using float division
        cell_width = self.size().width() / div
        cell_height = self.size().height() / div

        # Generate a list of characters from 'a' to 'z'
        alphabet = [chr(i) for i in range(ord('a'), ord('z') + 1)]

        for i in range(0, div):
            for j in range(0, div):
                # Calculate the top-left corner of each rectangle (i, j)
                x = i * cell_width
                y = j * cell_height
                
                # Draw the rectangle
                cyan = QColor(0, 255, 255)
                painter.setPen(QPen(cyan))
                cyan.setAlpha(20)
                painter.setBrush(QBrush(cyan))
                
                # Ensure integer values for drawing
                painter.drawRect(QRect(int(x), int(y), int(cell_width), int(cell_height)))

                # Create a label using characters instead of numbers
                label = f"{alphabet[i % 26]}{alphabet[j % 26]}"
                self.draw_element_label(painter, QPoint(int(x), int(y)), label)

    def draw_overlay_border(self, painter):
        pen = QPen(QColor(0, 255, 255))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawRect(self.rect().adjusted(2, 2, -2, -2))

    def draw_clickable_elements(self, painter):
        for element, label in zip(self.clickable_elements, self.element_labels):
            self.draw_element_label(painter, element.topLeft(), label)

            cyan = QColor(0, 255, 255)
            painter.setPen(QPen(cyan))
            cyan.setAlpha(20)
            painter.setBrush((QBrush(cyan)))

            painter.drawRect(element)

    def draw_element_label(self, painter, position, label):
        if not label:
            return

        x, y = position.x(), position.y()
        painter.setFont(self.label_font)
        metrics = QFontMetrics(self.label_font)

        text_width = metrics.width(label)
        font_height = metrics.height()

        padding_x = int(font_height * 0.1)
        padding_y = int(font_height * 0.1)

        rect_width = text_width + 2 * padding_x
        rect_height = font_height + 2 * padding_y

        # Draw background rectangle
        painter.fillRect(x, y, rect_width, rect_height, QColor(0, 255, 255))

        # Draw text
        painter.setPen(QColor(0, 0, 0))  # Black text color
        text_x = x + padding_x
        text_y = y + rect_height - padding_y - metrics.descent()
        painter.drawText(text_x, text_y, label)

        # Highlight matched part of the label
        if self.current_input:
            matched_text = label[:len(self.current_input)]
            matched_width = metrics.width(matched_text)
            painter.fillRect(x, y, matched_width + 2 * padding_x, rect_height, QColor(255, 255, 0, 128))  # Semi-transparent yellow
            painter.setPen(QColor(0, 0, 0))  # Black text color
            painter.drawText(text_x, text_y, matched_text)

    def draw_settings_info(self, painter):
        settings_width, settings_height = 105, 35
        painter.fillRect(0, 0, settings_width, settings_height, QColor(0, 0, 0))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(5, 15, "Confidence: 0.5")
        painter.drawText(5, 30, "Overlap: 0.7")
