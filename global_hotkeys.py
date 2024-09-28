from pynput import keyboard
from threading import Event

class GlobalHotKeys:
    def __init__(self):
        self.start_event = Event()
        self.stop_event = Event()
        self.exit_event = Event()
        self.listener = None
        self.alt_pressed = False

        # Define the hotkeys
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

    def on_release(self, key):
        print(f"Key released: {key}")
        if key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
            self.alt_pressed = False

    def start_listening(self):
        # Start the listener
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as l:
            self.listener = l  # Store listener reference
            l.join()

    def stop_listening(self):
        # Stop the listener if it exists
        if self.listener:
            self.listener.stop()

if __name__ == "__main__":
    global_hotkeys = GlobalHotKeys()
    global_hotkeys.start_listening()
