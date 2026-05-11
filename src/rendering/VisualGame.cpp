#include "VisualGame.hpp"
#include "../common/RNG.hpp"
#include <iostream>
#include <algorithm>
#include <cstdio>

namespace rendering {
    VisualGame::~VisualGame() { cleanup(); }

    bool VisualGame::init(bool ai, nn::PolicyNet* net, int steps, float eps, int episodes, float best) {
        ai_mode = ai;
        if (SDL_Init(SDL_INIT_VIDEO) < 0) { std::cerr << SDL_GetError() << "\n"; return false; }
        IMG_Init(IMG_INIT_JPG);
        TTF_Init();

        window   = SDL_CreateWindow("Spiral Shooter — A2C",
                                    SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                    constants::WIDTH, constants::HEIGHT, 0);
        renderer = SDL_CreateRenderer(window, -1,
                                      SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

        const char* fonts[] = {
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            nullptr
        };
        for (int i = 0; fonts[i]; i++) {
            font = TTF_OpenFont(fonts[i], 22);
            if (font) break;
        }

        sim = new game::GameSim(net);
        sim->step_count     = steps;
        sim->epsilon        = eps;
        sim->total_episodes = episodes;
        sim->best_score     = best;
        bg  = new Background(renderer);
        player_tex = IMG_LoadTexture(renderer, "Player.jpg");
        return true;
    }

    void VisualGame::run() {
        while (running) {
            handle_events();
            if (ai_mode) {
                bool alive = sim->step();
                if (!alive) {
                    spawn_death_particles(sim->player->get_pos());
                    sim->reset();
                }
            } else {
                human_step();
            }
            update_particles();
            render();
            SDL_Delay(16);
        }
    }

    void VisualGame::handle_events() {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) running = false;
            if (e.type == SDL_KEYDOWN) {
                keys[e.key.keysym.scancode] = 1;
                if (e.key.keysym.scancode == SDL_SCANCODE_ESCAPE) running = false;
            }
            if (e.type == SDL_KEYUP) keys[e.key.keysym.scancode] = 0;
            if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_LEFT)
                keys[SDL_SCANCODE_KP_0] = 1;
            if (e.type == SDL_MOUSEBUTTONUP && e.button.button == SDL_BUTTON_LEFT)
                keys[SDL_SCANCODE_KP_0] = 0;
        }
    }

    void VisualGame::human_step() {
        sim->player->update();
        Vec act(constants::NUM_ACTIONS, 0.f);
        if (keys[SDL_SCANCODE_W] || keys[SDL_SCANCODE_UP])    act[constants::MOVE_Y] -= 1.f;
        if (keys[SDL_SCANCODE_S] || keys[SDL_SCANCODE_DOWN])  act[constants::MOVE_Y] += 1.f;
        if (keys[SDL_SCANCODE_A] || keys[SDL_SCANCODE_LEFT])  act[constants::MOVE_X] -= 1.f;
        if (keys[SDL_SCANCODE_D] || keys[SDL_SCANCODE_RIGHT]) act[constants::MOVE_X] += 1.f;

        int mx, my;
        SDL_GetMouseState(&mx, &my);
        Vector2 aim_vec = (Vector2{(float)mx, (float)my} - sim->player->get_pos()).normalize();
        act[constants::AIM_ANGLE]  = std::atan2(aim_vec.y, aim_vec.x);
        if (keys[SDL_SCANCODE_SPACE] || keys[SDL_SCANCODE_KP_0]) act[constants::SHOOT_PROB] = 1.f;

        sim->player->move({act[constants::MOVE_X], act[constants::MOVE_Y]}, false);

        if (act[constants::SHOOT_PROB] > 0.5f && sim->player->can_shoot()) {
            Vector2 p_pos = sim->player->get_pos();
            sim->p_bullets.push_back(new game::LogicBullet(p_pos, p_pos + aim_vec * 1200.f, 10, 10.f));
            sim->player->fire();
            sim->shots_fired++;
        }

        int max_enemies    = std::min(8, 2 + sim->score/500);
        int spawn_interval = std::max(20, 80 - sim->score/300);
        if (++sim->spawn_timer >= spawn_interval && (int)sim->enemies.size() < max_enemies) {
            sim->enemies.push_back(new game::LogicEnemy(sim->player, sim->e_bullets));
            sim->spawn_timer = 0;
        }
        for (auto* e : sim->enemies) e->update(sim->score);
        for (auto* b : sim->p_bullets) b->update();
        for (auto* b : sim->e_bullets) b->update();

        for (auto eit = sim->enemies.begin(); eit != sim->enemies.end();) {
            game::Rect er = (*eit)->get_rect();
            bool killed = false;
            for (auto bit = sim->p_bullets.begin(); bit != sim->p_bullets.end();) {
                if (game::rects_intersect(er, (*bit)->get_rect())) {
                    (*eit)->take_damage(50);
                    sim->shots_hit++;
                    delete *bit; bit = sim->p_bullets.erase(bit);
                    if (!(*eit)->is_alive() && !killed) {
                        spawn_death_particles((*eit)->get_pos());
                        sim->score += 100; killed = true;
                    }
                } else ++bit;
            }
            if (killed) { delete *eit; eit = sim->enemies.erase(eit); continue; }
            if (game::rects_intersect((*eit)->get_rect(), sim->player->get_rect())) {
                std::cout << "[Human] Collided\n";
                sim->reset(); return;
            }
            ++eit;
        }
        for (auto bit = sim->e_bullets.begin(); bit != sim->e_bullets.end();) {
            if (!(*bit)->is_alive()) { delete *bit; bit = sim->e_bullets.erase(bit); continue; }
            if (game::rects_intersect((*bit)->get_rect(), sim->player->get_rect())) {
                sim->player->take_damage(40); shake = 16;
                delete *bit; bit = sim->e_bullets.erase(bit);
                if (sim->player->get_health() <= 0) { std::cout << "[Human] HP zero\n"; sim->reset(); return; }
            } else ++bit;
        }
        for (auto bit = sim->p_bullets.begin(); bit != sim->p_bullets.end();) {
            if (!(*bit)->is_alive()) { delete *bit; bit = sim->p_bullets.erase(bit); }
            else ++bit;
        }
    }

    void VisualGame::spawn_death_particles(Vector2 p) {
        SDL_Color colors[] = {{255,80,80,255},{255,160,60,255},{255,255,100,255}};
        for (int i = 0; i < 14; i++)
            particles.push_back({p, {RNG::flt(-3,3), RNG::flt(-3,3)}, 30, colors[RNG::rng_int(0,2)]});
    }

    void VisualGame::update_particles() {
        for (auto& p : particles) { p.pos += p.vel; p.life--; }
        particles.erase(std::remove_if(particles.begin(), particles.end(),
            [](const Particle& p){ return p.life <= 0; }), particles.end());
    }

    void VisualGame::draw_rect_entity(int ox, int oy, Vector2 pos, int w, int h, SDL_Color c) {
        SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
        SDL_SetRenderDrawColor(renderer, c.r, c.g, c.b, 60);
        SDL_Rect glow = { (int)(pos.x-w/2)-2+ox, (int)(pos.y-h/2)-2+oy, w+4, h+4 };
        SDL_RenderFillRect(renderer, &glow);
        SDL_SetRenderDrawColor(renderer, c.r, c.g, c.b, c.a);
        SDL_Rect rc = { (int)(pos.x-w/2)+ox, (int)(pos.y-h/2)+oy, w, h };
        SDL_RenderFillRect(renderer, &rc);
    }

    void VisualGame::render_text(const char* txt, int x, int y, SDL_Color c, bool center) {
        if (!font) return;
        SDL_Surface* surf = TTF_RenderText_Solid(font, txt, c);
        if (!surf) return;
        SDL_Texture* tex = SDL_CreateTextureFromSurface(renderer, surf);
        SDL_Rect dst = { x - (center ? surf->w/2 : 0), y, surf->w, surf->h };
        SDL_RenderCopy(renderer, tex, NULL, &dst);
        SDL_FreeSurface(surf);
        SDL_DestroyTexture(tex);
    }

    void VisualGame::draw_aim_indicator(int ox, int oy) {
        if (!sim) return;
        Vector2 p_pos = sim->player->get_pos();
        
        float min_ed = 1e9f;
        Vector2 near_ep = {0,0}, near_ev = {0,0};
        bool found_enemy = false;
        for (auto* e : sim->enemies) {
            if (!sim->is_visible(e->get_pos())) continue;
            float d = p_pos.distance_to(e->get_pos());
            if (d < min_ed) {
                min_ed = d;
                near_ep = e->get_pos();
                near_ev = e->get_vel_smooth();
                found_enemy = true;
            }
        }
        
        if (found_enemy) {
            Vector2 pred_pos = sim->predict_aim(p_pos, near_ep, near_ev, 10.f);
            Vector2 aim_dir = (pred_pos - p_pos).normalize();
            Vector2 tip = p_pos + aim_dir * 80.f;
            SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
            SDL_SetRenderDrawColor(renderer, 100, 255, 100, 200);
            SDL_RenderDrawLine(renderer, (int)p_pos.x+ox, (int)p_pos.y+oy, (int)tip.x+ox, (int)tip.y+oy);
            
            SDL_SetRenderDrawColor(renderer, 100, 255, 100, 100);
            SDL_Rect r = {(int)pred_pos.x+ox-4, (int)pred_pos.y+oy-4, 8, 8};
            SDL_RenderDrawRect(renderer, &r);
        }
    }

    void VisualGame::render() {
        int ox = 0, oy = 0;
        if (shake > 0) { ox = RNG::rng_int(-shake,shake); oy = RNG::rng_int(-shake,shake); shake--; }
        SDL_SetRenderDrawColor(renderer, 20, 20, 28, 255);
        SDL_RenderClear(renderer);
        bg->draw(renderer, ox, oy);
        if (ai_mode) {
            SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
            SDL_SetRenderDrawColor(renderer, 80, 80, 160, 60);
            int cx = constants::WIDTH/2, cy = constants::HEIGHT/2;
            SDL_Rect zone = {cx-300+ox, cy-300+oy, 600, 600};
            SDL_RenderDrawRect(renderer, &zone);
        }
        SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
        for (auto& p : particles) {
            SDL_SetRenderDrawColor(renderer, p.color.r, p.color.g, p.color.b, (uint8_t)(255*p.life/30));
            SDL_Rect rc = {(int)p.pos.x+ox-2, (int)p.pos.y+oy-2, 4, 4};
            SDL_RenderFillRect(renderer, &rc);
        }
        for (auto* b : sim->p_bullets)
            draw_rect_entity(ox, oy, b->get_pos(), 10, 10, {120, 255, 120, 255});
        for (auto* b : sim->e_bullets)
            draw_rect_entity(ox, oy, b->get_pos(), 8, 8, {255, 140, 80, 255});
        for (auto* e : sim->enemies) {
            draw_rect_entity(ox, oy, e->get_pos(), 26, 26, {255, 80, 80, 255});
            char buf[8]; sprintf(buf, "%d", e->get_health());
            render_text(buf, (int)e->get_pos().x+ox, (int)e->get_pos().y-32+oy, {255,100,100,255}, true);
        }
        if (player_tex) {
            Vector2 p = sim->player->get_pos();
            SDL_Rect dst = {(int)(p.x-14)+ox, (int)(p.y-14)+oy, 28, 28};
            SDL_RenderCopy(renderer, player_tex, NULL, &dst);
        } else {
            draw_rect_entity(ox, oy, sim->player->get_pos(), 28, 28, {200, 220, 255, 255});
        }
        if (ai_mode) draw_aim_indicator(ox, oy);
        draw_hud();
        SDL_RenderPresent(renderer);
    }

    void VisualGame::draw_hud() {
        if (!font) return;
        char buf[128];
        float acc = sim->shots_fired > 0 ? 100.f * sim->shots_hit / sim->shots_fired : 0.f;
        Vec V = sim->net->critic.forward(sim->get_state());
        SDL_Color hp_color = {200, 255, 200, 255};
        if      (sim->player->get_health() < 30) hp_color = {255, 80, 80, 255};
        else if (sim->player->get_health() < 60) hp_color = {255, 255, 100, 255};
        sprintf(buf, "HP: %d",          sim->player->get_health()); render_text(buf, 12, 12,  hp_color);
        sprintf(buf, "Score: %d",       sim->score);                render_text(buf, 12, 40,  {120, 200, 255, 255});
        sprintf(buf, "Reward: %.1f",    sim->total_reward);         render_text(buf, 12, 68,  {180, 240, 180, 255});
        sprintf(buf, "Acc: %.1f%%",     acc);                       render_text(buf, 12, 96,  {255, 220, 80, 255});
        sprintf(buf, "V_surv: %.2f",    V[0]);                      render_text(buf, 12, 124, {200, 200, 100, 255});
        sprintf(buf, "V_offn: %.2f",    V[1]);                      render_text(buf, 12, 152, {255, 150, 100, 255});
        sprintf(buf, "Eps: %.4f",       sim->epsilon);              render_text(buf, 12, 180, {160, 200, 255, 255});
        sprintf(buf, "Steps: %d",       sim->step_count);           render_text(buf, 12, 208, {160, 200, 255, 255});
        sprintf(buf, "Best: %d",        (int)sim->best_score);      render_text(buf, 12, 236, {255, 200, 100, 255});
        if (!ai_mode) render_text("HUMAN MODE", constants::WIDTH-160, 12, {255, 200, 80, 255});
        else          render_text("A2C MODE",   constants::WIDTH-130, 12, {80, 220, 255, 255});
    }

    void VisualGame::cleanup() {
        if (sim) {
            sim->finish_episode();
            sim->net->save("latest.weights", sim->step_count, sim->epsilon, sim->total_episodes, sim->best_score);
            sim->update_text_files();
            delete sim; sim = nullptr;
        }
        if (player_tex) { SDL_DestroyTexture(player_tex); player_tex = nullptr; }
        if (bg) { delete bg; bg = nullptr; }
        if (font) { TTF_CloseFont(font); font = nullptr; }
        if (renderer) { SDL_DestroyRenderer(renderer); renderer = nullptr; }
        if (window) { SDL_DestroyWindow(window); window = nullptr; }
        TTF_Quit(); IMG_Quit(); SDL_Quit();
    }
}
