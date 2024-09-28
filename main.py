import signal
import sys
from overlay_window import OverlayWindow
from PyQt5.QtWidgets import QApplication

# Create Qt application
app = QApplication(sys.argv)

# Allow Ctrl-C to stop application
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Create and start the overlay window
window = OverlayWindow()
window.render_start()

# Exit
sys.exit(app.exec_())
