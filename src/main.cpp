#include <QApplication>
#include <iostream>
#include <QScreen>
#include <QWindow>
#include <QWidget>
#include <QKeyEvent>
#include <QPainter>
#include <vector>

class OverlayWindow : public QWidget {
public:
    OverlayWindow(QScreen* screen) {
        setWindowFlags(Qt::FramelessWindowHint | 
                      Qt::WindowStaysOnTopHint |
                      Qt::Tool |
                      Qt::WindowDoesNotAcceptFocus);

        setAttribute(Qt::WA_TranslucentBackground);
        setAttribute(Qt::WA_TransparentForMouseEvents);
        setWindowFlags(Qt::X11BypassWindowManagerHint);
        setWindowState(Qt::WindowFullScreen);

        setGeometry(screen->geometry());
    }

protected:
    void keyPressEvent(QKeyEvent* event) override {
        if (event->key() == Qt::Key_Escape) {
            qApp->quit();
        }
    }

    void paintEvent(QPaintEvent* event) override {
        QPainter painter(this);
        painter.setPen(QPen(Qt::cyan, 3));
        painter.drawRect(rect());
    }
};

class OverlayManager {
public:
    OverlayManager() = default;

    void addOverlay(QScreen* screen) {
        auto window = new OverlayWindow(screen);
        windows.push_back(window);
    }

    void showOverlays() {
        for (auto* window : windows) {
            window->show();
        }
    }

    void hideOverlays() {
        for (auto* window : windows) {
            window->hide();
        }
    }

private:
    std::vector<OverlayWindow*> windows;
};

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);

    OverlayManager overlayManager;

    // Add a window for each screen
    for (QScreen* screen : QGuiApplication::screens()) {
        overlayManager.addOverlay(screen);
    }

    overlayManager.showOverlays();

    return app.exec();
}

