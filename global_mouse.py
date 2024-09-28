from pynput import mouse
from threading import Event

class GlobalMouse:
    def __init__(self):
        self.stop_event = Event()
        self.listener = None

    def on_move(self, x, y):
        pass

    def on_scroll(self, x, y, dx, dy):
        print("Mouse scrolled")
        self.stop_event.set()

    def on_click(self, x, y, button, pressed):
        # We no longer set the stop event on click
        pass

    def start_listening(self):
        with mouse.Listener(on_move=self.on_move, on_click=self.on_click, on_scroll=self.on_scroll) as listener:
            self.listener = listener
            listener.join()  # Wait until listener is stopped

    def stop_listening(self):
        if self.listener:
            self.listener.stop()

if __name__ == "__main__":
    global_mouse = GlobalMouse()
    global_mouse.start_listening()
