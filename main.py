import sys
import threading
from overlay_window import OverlayWindow
from global_hotkeys import GlobalHotKeys
from global_mouse import GlobalMouse
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer


def main():
    app = QApplication(sys.argv)

    overlay = OverlayWindow()
    overlay_thread = threading.Thread(target=overlay.activate_overlay)
    overlay_thread.start()

    hotkeys = GlobalHotKeys()
    hotkey_thread = threading.Thread(target=hotkeys.start_listening)
    hotkey_thread.start()

    mouse = GlobalMouse()
    mouse_thread = threading.Thread(target=mouse.start_listening)
    mouse_thread.start()

    # Connect signals
    hotkeys.key_pressed_signal.connect(overlay.key_pressed_signal)
    hotkeys.grid_view_signal.connect(overlay.toggle_grid_view)
    hotkeys.stop_grid_view_signal.connect(overlay.stop_grid_view)

    def check_events():
        if hotkeys.start_event.is_set():
            overlay.start_element_detection()
            hotkeys.set_detection_active(True)
            hotkeys.start_event.clear()
        
        if hotkeys.stop_event.is_set():
            overlay.stop_element_detection()
            overlay.stop_grid_view()  # Also stop grid view when ESC is pressed
            hotkeys.set_detection_active(False)
            hotkeys.is_grid_view_active = False
            hotkeys.stop_event.clear()
        
        if mouse.stop_event.is_set():
            overlay.stop_element_detection()
            overlay.stop_grid_view()  # Also stop grid view on mouse click
            hotkeys.set_detection_active(False)
            hotkeys.is_grid_view_active = False
            mouse.stop_event.clear()

        if hotkeys.exit_event.is_set():
            app.quit()

    timer = QTimer()
    timer.timeout.connect(check_events)
    timer.start(10)

    exit_code = app.exec_()

    hotkeys.stop_listening()
    hotkey_thread.join()

    mouse.stop_listening()
    mouse_thread.join()

    sys.exit(exit_code)

if __name__ == "__main__":
    main()
