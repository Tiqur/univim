from pynput import keyboard
from threading import Event

class GlobalHotKeys:
    def __init__(self):
        self.start_event = Event()
        self.stop_event = Event()
        self.exit_event = Event()
        self.listener = None

        # Define the hotkeys
        self.hotkeys = [
            keyboard.HotKey(keyboard.HotKey.parse('<alt>+f'), self.on_activate_f)
        ]

    def for_canonical(self, f):
        return lambda k: f(self.listener.canonical(k))

    #def on_activate_esc(self):
    #    self.stop_event.set()

    def on_activate_f(self):
        self.start_event.set()

    def on_press(self, key):
        print(f"Key pressed: {key}")
        # Press handler for all hotkeys
        for hotkey in self.hotkeys:
            hotkey.press(self.listener.canonical(key))

    def on_release(self, key):
        # Release handler for all hotkeys
        for hotkey in self.hotkeys:
            hotkey.release(self.listener.canonical(key))

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

