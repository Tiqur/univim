g++ src/main.cpp $(pkg-config --cflags --libs Qt6Core Qt6Widgets Qt6Gui) -fPIC && ./a.out
