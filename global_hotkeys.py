from pynput import keyboard
from threading import Event
from PyQt5.QtCore import QObject, pyqtSignal

class GlobalHotKeys(QObject):
    key_pressed_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.start_event = Event()
        self.stop_event = Event()
        self.exit_event = Event()
        self.listener = None
        self.alt_pressed = False
        self.is_detection_active = False

        self.hotkeys = {
            keyboard.KeyCode.from_char('f'): self.on_activate_f,
            keyboard.Key.esc: self.on_activate_esc,
            keyboard.Key.alt_l: self.on_alt,
            keyboard.Key.alt_r: self.on_alt
        }

    def on_activate_esc(self):
        print("ESC key pressed")
        self.stop_event.set()

    def on_activate_f(self):
        if self.alt_pressed:
            print("Alt+F key pressed")
            self.start_event.set()

    def on_alt(self):
        self.alt_pressed = True

    def on_press(self, key):
        print(f"Key pressed: {key}")
        if key in self.hotkeys:
            self.hotkeys[key]()
        elif self.is_detection_active:
            try:
                char = key.char.lower()
                self.key_pressed_signal.emit(char)
            except AttributeError:
                pass  # Ignore special keys

    def on_release(self, key):
        print(f"Key released: {key}")
        if key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
            self.alt_pressed = False

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
