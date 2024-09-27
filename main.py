import signal
import sys
import threading
from overlay_window import OverlayWindow
from PyQt5.QtWidgets import QApplication
from screenshot import capture_screenshot
from PyQt5.QtCore import QRect
from PyQt5.QtGui import QColor

capture_screenshot()

# Create Qt application
app = QApplication(sys.argv)

# Allow Ctrl-C to stop application
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Display window
window = OverlayWindow()

from ultralytics import YOLO
model = YOLO("weights/best.pt")
results = model("screenshot.png", conf=0.1, max_det=2048, iou=0.4, stream=True)

def is_significant_overlap(box1, box2, threshold=0.2):
    intersect = box1.intersected(box2)
    intersect_area = intersect.width() * intersect.height()
    smaller_box_area = min(box1.width() * box1.height(), box2.width() * box2.height())
    return intersect_area / smaller_box_area > threshold

def filter_boxes(boxes):
    # Sort boxes by area in descending order
    sorted_boxes = sorted(boxes, key=lambda b: b.width() * b.height())
    
    filtered_boxes = []
    for i, box in enumerate(sorted_boxes):
        should_add = True
        for j in range(i):
            if is_significant_overlap(box, sorted_boxes[j]):
                should_add = False
                break
        if should_add:
            filtered_boxes.append(box)
    
    return filtered_boxes

def secondary_thread():
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

        filtered_boxes = filter_boxes(all_boxes)
        
        for box in filtered_boxes:
            window.add_box(box)

        window.send_update_signal()

x = threading.Thread(target=secondary_thread)
x.start()

window.show()

# Exit
sys.exit(app.exec_())
