from pynput import keyboard
from PyQt5.QtCore import QObject, pyqtSignal
from threading import Event
import time

class GlobalHotKeys(QObject):
    key_pressed_signal = pyqtSignal(str)
    grid_view_signal = pyqtSignal()
    stop_grid_view_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.start_event = Event()
        self.stop_event = Event()
        self.exit_event = Event()
        self.scroll_up_event = Event()
        self.scroll_down_event = Event()
        self.grid_view_event = Event()
        self.listener = None
        self.is_detection_active = False
        self.is_grid_view_active = False
        
        # Variables to track the last pressed time
        self.last_shift_press_time = 0
        self.last_ctrl_press_time = 0
        self.double_press_threshold = 0.3  # Time in seconds for double press detection

        self.hotkeys = {
            keyboard.Key.shift: self.on_shift,
            keyboard.Key.ctrl: self.on_ctrl,
            keyboard.Key.esc: self.on_activate_esc
        }

    def on_activate_esc(self):
        print("ESC key pressed")
        if self.is_grid_view_active:
            self.stop_grid_view_signal.emit()
            self.is_grid_view_active = False
        else:
            self.stop_event.set()

    def on_shift(self):
        current_time = time.time()
        # Check if the time since the last press is within the threshold
        if current_time - self.last_shift_press_time <= self.double_press_threshold:
            print("Shift 2nd pressed")
            self.start_event.set()
        else:
            print("Shift key pressed")
        
        # Update the last pressed time
        self.last_shift_press_time = current_time

    def on_ctrl(self):
        current_time = time.time()
        # Check if the time since the last press is within the threshold
        if current_time - self.last_ctrl_press_time <= self.double_press_threshold:
            print("Ctrl 2nd pressed")
            self.is_grid_view_active = not self.is_grid_view_active
            self.grid_view_signal.emit()
        else:
            print("Ctrl key pressed")
        
        # Update the last pressed time
        self.last_ctrl_press_time = current_time

    def on_press(self, key):
        print(f"Key pressed: {key}")
        if key in self.hotkeys:
            self.hotkeys[key]()
        elif self.is_detection_active or self.is_grid_view_active:
            try:
                char = key.char.lower()
                print(f"Emitting key pressed signal: {char}")
                self.key_pressed_signal.emit(char)
            except AttributeError:
                pass

    def on_release(self, key):
        print(f"Key released: {key}")

    def start_listening(self):
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as l:
            self.listener = l
            l.join()

    def stop_listening(self):
        if self.listener:
            self.listener.stop()

    def set_detection_active(self, active):
        self.is_detection_active = active

if __name__ == "__main__":
    global_hotkeys = GlobalHotKeys()
    global_hotkeys.start_listening()

