import signal
import sys
import random
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
# In the future, use stream for real-time inferences https://docs.ultralytics.com/modes/predict/#inference-sources
# OR use "screen" to capture screen without taking a screenshot
results = model("screenshot.png")

for result in results:
    for box in result.boxes:
        dim = (box.xywhn).tolist()[0]
        orig_width = box.orig_shape[1]
        orig_height = box.orig_shape[0]
        print("BOX: ", dim, orig_width, orig_height)

        width = int(dim[2]*orig_width)
        height = int(dim[3]*orig_height)
        x = int(dim[0]*orig_width-width/2)
        y = int(dim[1]*orig_height-height/2)

        box = Box(x, y, width, height)
        window.boxes.append(box)


window.show()

# Exit
sys.exit(app.exec_())
