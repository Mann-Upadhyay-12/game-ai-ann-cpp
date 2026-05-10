CXX = g++
CXXFLAGS = -Wall -O3 -g -std=c++17
SDL_CFLAGS = `sdl2-config --cflags`
SDL_LDFLAGS = `sdl2-config --libs` -lSDL2_image -lSDL2_ttf

SRC_DIR = src
OBJ_DIR = obj

COMMON_SRCS = $(SRC_DIR)/nn/ValueNet.cpp \
              $(SRC_DIR)/nn/PolicyNet.cpp \
              $(SRC_DIR)/game/Entities.cpp \
              $(SRC_DIR)/game/GameSim.cpp \
              $(SRC_DIR)/main.cpp

GUI_SRCS = $(SRC_DIR)/rendering/Background.cpp \
           $(SRC_DIR)/rendering/VisualGame.cpp

TARGET = shooter
HEADLESS_TARGET = shooter_headless

GUI_OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/gui/%.o, $(COMMON_SRCS) $(GUI_SRCS))
HEADLESS_OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/headless/%.o, $(COMMON_SRCS))

all: $(TARGET)

$(TARGET): $(GUI_OBJS)
	$(CXX) $(GUI_OBJS) -o $(TARGET) $(SDL_LDFLAGS)

$(HEADLESS_TARGET): $(HEADLESS_OBJS)
	$(CXX) $(HEADLESS_OBJS) -o $(HEADLESS_TARGET)

$(OBJ_DIR)/gui/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(SDL_CFLAGS) -c $< -o $@

$(OBJ_DIR)/headless/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -DHEADLESS -c $< -o $@

headless: $(HEADLESS_TARGET)

clean:
	rm -rf $(OBJ_DIR) $(TARGET) $(HEADLESS_TARGET)

.PHONY: all headless clean
