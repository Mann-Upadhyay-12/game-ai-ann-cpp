#pragma once
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include "../common/Constants.hpp"

namespace rendering {
    class Background {
        SDL_Texture* tex = nullptr;
        int bw = 0, bh = 0;
        float scroll = 0;
    public:
        Background(SDL_Renderer* r);
        ~Background();
        void update();
        void draw(SDL_Renderer* r, int ox, int oy);
    };
}
