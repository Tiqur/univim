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

for _ in range(100):
    x = random.randint(0, 2440)
    y = random.randint(0, 1440)
    width = random.randint(20, 100)
    height = random.randint(20, 100)

    box = Box(x, y, width, height)
    window.boxes.append(box)


window.show()

# Exit
sys.exit(app.exec_())


from ultralytics import YOLO

model = YOLO("weights/best.pt")
results = model("screenshot.png")

for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.save(filename="result.jpg")  # save to disk
