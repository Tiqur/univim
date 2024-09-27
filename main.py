import signal
import sys
import random
import threading
from overlay_window import OverlayWindow, Box
from PyQt5.QtWidgets import QApplication
from screenshot import capture_screenshot

capture_screenshot()

# Create Qt application
app = QApplication(sys.argv)

# Allow Ctrl-C to stop application
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Display window
window = OverlayWindow()

from ultralytics import YOLO
model = YOLO("weights/best.pt")
results = model("screen", conf=0.1, max_det=2048, iou=0.4, stream=True)

def secondary_thread():
    for result in results:

        # Add boxes to window
        for box in result.boxes:
            dim = (box.xywhn).tolist()[0]
            orig_width = box.orig_shape[1]
            orig_height = box.orig_shape[0]
            print("BOX: ", dim, orig_width, orig_height)

            width = int(dim[2]*orig_width)
            height = int(dim[3]*orig_height)
            x = int(dim[0]*orig_width-width/2)
            y = int(dim[1]*orig_height-height/2)

            new_box = Box(x, y, width, height)
            window.add_box(new_box)

x = threading.Thread(target=secondary_thread)
x.start()

window.show()

# Exit
sys.exit(app.exec_())
