import signal
import sys
from overlay_window import OverlayWindow
from PyQt5.QtWidgets import QApplication
from screenshot import capture_screenshot

capture_screenshot()

# Create Qt application
app = QApplication(sys.argv)

# Allow Ctrl-C to stop application
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Display window
window = OverlayWindow()
window.show()

# Exit
sys.exit(app.exec_())


