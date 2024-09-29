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
        self.grid_mode = 'main'  # 'main' or 'zoomed'
        self.selected_cell = None
        self.zoom_factor = 2
        self.zoomed_image = None
        self.zoom_height_percentage = 0.2
        self.zoomed_rect = None
        self.subgrid_divisions = 6
        self.full_screenshot = None

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
        print(f"Key pressed: {key}, Current mode: {self.grid_mode}, Current input: {self.current_input}")
        if self.is_grid_view_active:
            if key.isalpha() or key.isdigit():
                self.current_input += key.lower()
                print(f"Updated input: {self.current_input}")
                if self.grid_mode == 'main':
                    if len(self.current_input) == 2:
                        self.select_main_cell(self.current_input)
                elif self.grid_mode == 'zoomed':
                    if len(self.current_input) == 1:
                        self.select_subcell(self.current_input)
            elif key == 'backspace':
                self.current_input = self.current_input[:-1]
            elif key == 'esc':
                self.stop_grid_view()
            self.update()
        elif self.is_overlay_active:
            if key.isalpha():
                self.current_input += key.lower()
                self.update_labels_starting_with(self.current_input)
            elif key == 'backspace':
                self.current_input = self.current_input[:-1]
                self.update_labels_starting_with(self.current_input)
            elif key == 'esc':
                self.stop_element_detection()
        self.update()

    def select_main_cell(self, cell_id):
        print(f"Selecting main cell: {cell_id}")
        if len(cell_id) == 2 and cell_id.isalpha():
            row = ord(cell_id[0]) - ord('a')
            col = ord(cell_id[1]) - ord('a')
            if 0 <= row < 26 and 0 <= col < 26:
                self.selected_cell = (row, col)
                self.grid_mode = 'zoomed'
                self.current_input = ''
                self.capture_zoomed_cell()
                print(f"Selected cell: {self.selected_cell}, Switching to zoomed mode")
            else:
                print("Invalid cell selection")
        else:
            print("Invalid cell_id format")
        self.update()


    def capture_zoomed_cell(self):
        if self.selected_cell and self.full_screenshot:
            row, col = self.selected_cell
            cell_width = self.width() / 26
            cell_height = self.height() / 26
            
            x = int(col * cell_width)
            y = int(row * cell_height)
            
            self.zoomed_image = self.full_screenshot.copy(x, y, int(cell_width), int(cell_height))



    def select_subcell(self, subcell_id):
        if self.selected_cell and self.grid_mode == 'zoomed' and self.zoomed_rect:
            if subcell_id.isalpha():
                sub_index = ord(subcell_id.lower()) - ord('a')
            elif subcell_id.isdigit():
                sub_index = 26 + int(subcell_id)
            else:
                return  # Invalid input

            if 0 <= sub_index < self.subgrid_divisions ** 2:
                sub_row = sub_index // self.subgrid_divisions
                sub_col = sub_index % self.subgrid_divisions

                # Calculate the position within the zoomed image
                cell_width = self.zoomed_rect.width() / self.subgrid_divisions
                cell_height = self.zoomed_rect.height() / self.subgrid_divisions
                
                relative_x = (sub_col + 0.5) * cell_width
                relative_y = (sub_row + 0.5) * cell_height

                # Calculate the position relative to the zoomed area
                zoomed_x = relative_x / self.zoomed_rect.width()
                zoomed_y = relative_y / self.zoomed_rect.height()

                # Translate to actual screen coordinates
                main_row, main_col = self.selected_cell
                main_cell_width = self.width() / 26
                main_cell_height = self.height() / 26

                screen_x = (main_col * main_cell_width) + (zoomed_x * main_cell_width)
                screen_y = (main_row * main_cell_height) + (zoomed_y * main_cell_height)

                # Perform the click
                self.mouse.position = (int(screen_x), int(screen_y))
                self.mouse.click(Button.left)
                print(f"Clicked at {screen_x}, {screen_y}")
                
                # Reset the view
                self.is_grid_view_active = False
                self.grid_mode = 'main'
                self.selected_cell = None
                self.current_input = ''
                self.zoomed_image = None
                self.zoomed_rect = None
        self.update()


    def capture_full_screenshot(self):
        screen = QApplication.primaryScreen()
        self.full_screenshot = screen.grabWindow(0)

    def toggle_grid_view(self):
        if not self.is_grid_view_active:
            self.capture_full_screenshot()
        
        self.is_grid_view_active = not self.is_grid_view_active
        self.is_overlay_active = False
        self.grid_mode = 'main'
        self.current_input = ''
        self.selected_cell = None
        print(f"Grid view {'activated' if self.is_grid_view_active else 'deactivated'}")
        self.update()

    def stop_grid_view(self):
        print("Stopping grid view")  
        self.is_grid_view_active = False
        self.grid_mode = 'main'
        self.selected_cell = None
        self.current_input = ''
        self.update()

    def update_labels_starting_with(self, input_string):
        matching_labels = [label for label in self.element_labels if label.lower().startswith(input_string.lower())]
        
        if not matching_labels:
            # If no matches, keep all labels visible
            self.current_input = ''
            return

        # Update visible labels for matches
        for i, label in enumerate(self.element_labels):
            if label.lower().startswith(input_string.lower()):
                self.element_labels[i] = label
            else:
                self.element_labels[i] = ''  # Hide non-matching labels

        # If there's only one match and it's exactly the input, trigger the click
        if len(matching_labels) == 1 and matching_labels[0].lower() == input_string.lower():
            index = self.element_labels.index(matching_labels[0])
            self.click_element(index)
            self.current_input = ''  # Reset input after clicking

        self.update()

    def scroll_up(self):
        self.mouse.scroll(0, 2)

    def scroll_down(self):
        self.mouse.scroll(0, -2)

    def click_element(self, index):
        if 0 <= index < len(self.clickable_elements):
            element = self.clickable_elements[index]
            center = element.center()
            self.mouse.position = (center.x(), center.y())
            self.mouse.click(Button.left)
            print(f"Clicked element at {center.x()}, {center.y()}")
            
            # Reset the overlay after clicking
            self.stop_element_detection()
        else:
            print(f"Invalid index: {index}")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), Qt.transparent)

        if not self.is_overlay_active and not self.is_grid_view_active:
            return

        self.draw_overlay_border(painter)
        
        if self.is_overlay_active:
            self.draw_clickable_elements(painter)
        
        if self.is_grid_view_active:
            if self.grid_mode == 'main':
                self.draw_main_grid(painter)
            elif self.grid_mode == 'zoomed':
                self.draw_zoomed_cell(painter)

    def draw_zoomed_cell(self, painter):
        if not self.zoomed_image or not self.selected_cell:
            return

        original_aspect_ratio = self.zoomed_image.width() / self.zoomed_image.height()
        zoom_height = self.height() * self.zoom_height_percentage
        zoom_width = zoom_height * original_aspect_ratio

        # Calculate the position of the selected cell
        cell_width = self.width() / 26
        cell_height = self.height() / 26
        selected_x = self.selected_cell[1] * cell_width
        selected_y = self.selected_cell[0] * cell_height

        # Calculate the position for the zoomed image
        x = max(0, min(selected_x - zoom_width / 2, self.width() - zoom_width))
        y = max(0, min(selected_y - zoom_height / 2, self.height() - zoom_height))

        self.zoomed_rect = QRect(int(x), int(y), int(zoom_width), int(zoom_height))

        painter.fillRect(self.rect(), QColor(0, 0, 0, 128))
        painter.drawPixmap(self.zoomed_rect, self.zoomed_image)

        pen = QPen(QColor(0, 255, 255))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawRect(self.zoomed_rect)

        self.draw_zoomed_grid(painter)



    def draw_zoomed_grid(self, painter):
        if not self.zoomed_rect:
            return

        cell_width = self.zoomed_rect.width() / self.subgrid_divisions
        cell_height = self.zoomed_rect.height() / self.subgrid_divisions

        pen = QPen(QColor(0, 255, 255))
        pen.setWidth(1)
        painter.setPen(pen)

        for i in range(1, self.subgrid_divisions):
            x = self.zoomed_rect.left() + i * cell_width
            painter.drawLine(int(x), self.zoomed_rect.top(), int(x), self.zoomed_rect.bottom())

            y = self.zoomed_rect.top() + i * cell_height
            painter.drawLine(self.zoomed_rect.left(), int(y), self.zoomed_rect.right(), int(y))

        for i in range(self.subgrid_divisions):
            for j in range(self.subgrid_divisions):
                x = self.zoomed_rect.left() + j * cell_width
                y = self.zoomed_rect.top() + i * cell_height
                
                cyan = QColor(0, 255, 255)
                painter.setPen(QPen(cyan))
                cyan.setAlpha(20)
                painter.setBrush(QBrush(cyan))
                
                painter.drawRect(QRect(int(x), int(y), int(cell_width), int(cell_height)))

                cell_index = i * self.subgrid_divisions + j
                if cell_index < 26:
                    label = chr(97 + cell_index)  # 'a' to 'z'
                elif cell_index < 36:
                    label = str(cell_index - 26)  # '0' to '9'
                else:
                    continue

                if not self.current_input or label.startswith(self.current_input):
                    self.draw_element_label(painter, QPoint(int(x), int(y)), label)

    def draw_main_grid(self, painter):
        pen = QPen(QColor(0, 255, 255))
        pen.setWidth(2)
        painter.setPen(pen)

        cell_width = self.width() / 26
        cell_height = self.height() / 26

        for i in range(26):
            for j in range(26):
                x = i * cell_width
                y = j * cell_height
                
                cyan = QColor(0, 255, 255)
                painter.setPen(QPen(cyan))
                cyan.setAlpha(20)
                painter.setBrush(QBrush(cyan))
                
                painter.drawRect(QRect(int(x), int(y), int(cell_width), int(cell_height)))

                label = f"{chr(97 + j)}{chr(97 + i)}"
                if not self.current_input or label.startswith(self.current_input):
                    self.draw_element_label(painter, QPoint(int(x), int(y)), label)

    def draw_sub_grid(self, painter):
        if not self.selected_cell:
            return

        main_row, main_col = self.selected_cell
        cell_width = self.width() / 26
        cell_height = self.height() / 26

        painter.setFont(self.label_font)
        metrics = QFontMetrics(self.label_font)
        char_width = metrics.horizontalAdvance('W')  # Use 'W' as a reference for max width
        char_height = metrics.height()

        subcell_width = max(cell_width / self.subcell_divisions, char_width * 1.5)
        subcell_height = max(cell_height / self.subcell_divisions, char_height * 1.5)

        for i in range(self.subcell_divisions):
            for j in range(self.subcell_divisions):
                x = (main_col * cell_width) + (j * subcell_width)
                y = (main_row * cell_height) + (i * subcell_height)
                
                # Semi-transparent background
                cyan = QColor(0, 255, 255, 40)  # Increased transparency
                painter.setBrush(QBrush(cyan))
                painter.setPen(Qt.NoPen)  # No border
                
                painter.drawRect(QRect(int(x), int(y), int(subcell_width), int(subcell_height)))

                # Determine label
                index = i * self.subcell_divisions + j
                if index < 26:
                    label = chr(97 + index)  # 'a' to 'z'
                elif index < 36:
                    label = str(index - 26)  # '0' to '9'
                else:
                    continue  # Skip if we've used all labels

                # Draw label
                painter.setPen(QColor(0, 0, 0))  # Black text
                painter.drawText(QRect(int(x), int(y), int(subcell_width), int(subcell_height)), 
                                 Qt.AlignCenter, label)

    def draw_overlay_border(self, painter):
        pen = QPen(QColor(0, 255, 255))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawRect(self.rect().adjusted(2, 2, -2, -2))


    def draw_clickable_elements(self, painter):
        for element, label in zip(self.clickable_elements, self.element_labels):
            if label:  # Only draw elements with non-empty labels
                self.draw_element_label(painter, element.topLeft(), label)

                cyan = QColor(0, 255, 255)
                painter.setPen(QPen(cyan))
                cyan.setAlpha(20)
                painter.setBrush(QBrush(cyan))

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
            painter.fillRect(x, y, matched_width + 2 * padding_x, rect_height, QColor(0, 255, 0, 128))  # Semi-transparent green
            painter.setPen(QColor(0, 0, 0))  # Black text color
            painter.drawText(text_x, text_y, matched_text)


    def draw_settings_info(self, painter):
        settings_width, settings_height = 105, 35
        painter.fillRect(0, 0, settings_width, settings_height, QColor(0, 0, 0))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(5, 15, "Confidence: 0.5")
        painter.drawText(5, 30, "Overlap: 0.7")
