import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QScreen
from PyQt5.QtCore import QDateTime


def capture_screenshot():
    app = QApplication(sys.argv)
    
    # Get the primary screen
    screen = QApplication.primaryScreen()

    # Take a screenshot
    screenshot = screen.grabWindow(0)

    # Save the screenshot to a file
    screenshot.save('screenshot.png', 'png')
