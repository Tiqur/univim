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
    current_time = QDateTime.currentDateTime().toString("yyyyMMdd_HHmmss")
    file_name = f"screenshot_{current_time}.png"
    screenshot.save(file_name, 'png')

    print(f"Screenshot saved as {file_name}")
