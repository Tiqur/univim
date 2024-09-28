import sys
import threading
from overlay_window import OverlayWindow
from global_hotkeys import GlobalHotKeys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

def main():
    # Create Qt application
    app = QApplication(sys.argv)

    # Create the overlay window
    window = OverlayWindow()

    # Start the hotkey listener in a separate thread
    window_thread = threading.Thread(target=window.start)
    window_thread.start()

    # Create the global hotkey listener
    hotkeys = GlobalHotKeys()

    # Start the hotkey listener in a separate thread
    hotkey_thread = threading.Thread(target=hotkeys.start_listening)
    hotkey_thread.start()

    # Function to check events and update the window
    def check_events():
        if hotkeys.start_event.is_set():
            window.render_start()
            hotkeys.start_event.clear()
        if hotkeys.stop_event.is_set():
            window.render_stop()
            hotkeys.stop_event.clear()
        if hotkeys.exit_event.is_set():
            app.quit()

    # Set up a timer to check for events
    timer = QTimer()
    timer.timeout.connect(check_events)
    timer.start(10)  # Check every 100ms

    # Run the application
    exit_code = app.exec_()

    # Clean up
    hotkeys.stop_listening()
    hotkey_thread.join()

    sys.exit(exit_code)

if __name__ == "__main__":
    main()
