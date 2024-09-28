import sys
import threading
from overlay_window import OverlayWindow
from global_hotkeys import GlobalHotKeys
from global_mouse import GlobalMouse
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

def main():
    # Create Qt application
    app = QApplication(sys.argv)

    # Create the OverlayWindow
    overlay = OverlayWindow()

    # Activate the overlay in a separate thread
    overlay_thread = threading.Thread(target=overlay.activate_overlay)
    overlay_thread.start()

    # Create the global hotkey listener and start in a separate thread
    hotkeys = GlobalHotKeys()
    hotkey_thread = threading.Thread(target=hotkeys.start_listening)
    hotkey_thread.start()

    # Create the global mouse listener and start in a separate thread
    mouse = GlobalMouse()
    mouse_thread = threading.Thread(target=mouse.start_listening)
    mouse_thread.start()

    # Function to check events and update the overlay
    def check_events():
        if hotkeys.start_event.is_set():
            overlay.start_element_detection()
            hotkeys.start_event.clear()  # Clear after starting detection
        
        if hotkeys.stop_event.is_set():
            overlay.stop_element_detection()
            hotkeys.stop_event.clear()  # Clear after stopping detection
        
        if mouse.stop_event.is_set():
            overlay.stop_element_detection()
            mouse.stop_event.clear()  # Clear mouse stop event

        if hotkeys.exit_event.is_set():
            app.quit()

    # Set up a timer to check for events
    timer = QTimer()
    timer.timeout.connect(check_events)
    timer.start(10)  # Check every 10ms

    # Run the application
    exit_code = app.exec_()

    # Clean up
    hotkeys.stop_listening()
    hotkey_thread.join()

    sys.exit(exit_code)

if __name__ == "__main__":
    main()
