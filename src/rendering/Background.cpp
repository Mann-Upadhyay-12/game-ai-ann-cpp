#include "Background.hpp"

namespace rendering {
    Background::Background(SDL_Renderer* r) {
        tex = IMG_LoadTexture(r, "back.jpg");
        if (tex) {
            SDL_QueryTexture(tex, NULL, NULL, &bw, &bh);
            SDL_SetTextureColorMod(tex, 120, 120, 160);
        }
    }
    Background::~Background() { if (tex) SDL_DestroyTexture(tex); }
    void Background::update() { scroll += 0.8f; if (scroll >= bw) scroll = 0; }
    void Background::draw(SDL_Renderer* r, int ox, int oy) {
        if (!tex) return;
        for (float x = -scroll; x < constants::WIDTH; x += bw) {
            SDL_Rect dst = {(int)x + ox, oy, bw, constants::HEIGHT};
            SDL_RenderCopy(r, tex, NULL, &dst);
        }
    }
}
