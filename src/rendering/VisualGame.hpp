#pragma once
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_ttf.h>
#include "../game/GameSim.hpp"
#include "Background.hpp"
#include <vector>
#include <cstdint>

namespace rendering {
    struct Particle {
        Vector2 pos, vel;
        int life;
        SDL_Color color;
    };

    class VisualGame {
        SDL_Window*   window   = nullptr;
        SDL_Renderer* renderer = nullptr;
        SDL_Texture*  player_tex = nullptr;
        TTF_Font*     font     = nullptr;
        Background*   bg       = nullptr;

        game::GameSim* sim  = nullptr;
        bool     running = true;
        bool     ai_mode;
        int      shake = 0;

        uint8_t keys[SDL_NUM_SCANCODES] = {};
        std::vector<Particle> particles;

    public:
        ~VisualGame();
        bool init(bool ai, nn::PolicyNet* net, int steps, float eps, int episodes, float best);
        void run();

    private:
        void handle_events();
        void human_step();
        void spawn_death_particles(Vector2 p);
        void update_particles();
        void draw_rect_entity(int ox, int oy, Vector2 pos, int w, int h, SDL_Color c);
        void render_text(const char* txt, int x, int y, SDL_Color c, bool center = false);
        void draw_aim_indicator(int ox, int oy);
        void render();
        void draw_hud();
        void cleanup();
    };
}
