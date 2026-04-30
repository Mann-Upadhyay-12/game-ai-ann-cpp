CXX = g++
CXXFLAGS = -Wall -O2 `sdl2-config --cflags`
LDFLAGS = `sdl2-config --libs` -lSDL2_image -lSDL2_ttf

all: shooter

shooter: main.cpp
	$(CXX) main.cpp -o shooter $(CXXFLAGS) $(LDFLAGS)

clean:
	rm -f shooter
