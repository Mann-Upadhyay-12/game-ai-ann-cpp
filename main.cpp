#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_ttf.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

// Constants
const int WIDTH = 1200;
const int HEIGHT = 900;
const int CUBE_SIZE = 30;

// Enums
enum Action {
    IDLE,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    SHOOT
};

// Math Utilities
class Vector2 {
public:
    float x, y;

    Vector2(float x = 0, float y = 0) : x(x), y(y) {}

    Vector2 operator+(const Vector2& v) const { return {x + v.x, y + v.y}; }
    Vector2 operator-(const Vector2& v) const { return {x - v.x, y - v.y}; }
    Vector2 operator*(float s) const { return {x * s, y * s}; }
    Vector2 operator/(float s) const { return {x / s, y / s}; }
    Vector2& operator+=(const Vector2& v) { x += v.x; y += v.y; return *this; }
    
    float length() const { return std::sqrt(x * x + y * y); }
    
    Vector2 normalize() const {
        float l = length();
        if (l > 0) return {x / l, y / l};
        return {0, 0};
    }
    
    float distance_to(const Vector2& v) const {
        return (*this - v).length();
    }
};

class Random {
public:
    static float get_float(float min, float max) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(min, max);
        return dis(gen);
    }

    static int get_int(int min, int max) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(min, max);
        return dis(gen);
    }
};

// Base Classes
class Entity {
protected:
    Vector2 pos;
    SDL_Rect rect;
    SDL_Color color;
    bool alive;

public:
    Entity(Vector2 p, int w, int h, SDL_Color c) : pos(p), color(c), alive(true) {
        rect = { 0, 0, w, h };
    }
    
    virtual ~Entity() {}

    virtual void update() = 0;
    
    virtual void draw(SDL_Renderer* renderer, int ox, int oy) {
        SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);
        SDL_Rect r = { 
            (int)std::round(pos.x - rect.w / 2) + ox, 
            (int)std::round(pos.y - rect.h / 2) + oy, 
            rect.w, 
            rect.h 
        };
        SDL_RenderFillRect(renderer, &r);
    }

    bool is_alive() const { return alive; }
    void die() { alive = false; }
    Vector2 get_pos() const { return pos; }
    SDL_Rect get_rect() const {
        return { (int)(pos.x - rect.w / 2), (int)(pos.y - rect.h / 2), rect.w, rect.h };
    }
};

// Game Entities
class Particle : public Entity {
private:
    Vector2 velocity;
    int lifetime;

public:
    Particle(Vector2 p, SDL_Color c) 
        : Entity(p, 4, 4, c), 
          velocity(Random::get_float(-3, 3), Random::get_float(-3, 3)), 
          lifetime(30) {}

    void update() override {
        pos += velocity;
        lifetime--;
        if (lifetime <= 0) die();
    }
};

class Bullet : public Entity {
private:
    Vector2 velocity;

public:
    Bullet(Vector2 p, Vector2 target, SDL_Color c, int size, float speed) 
        : Entity(p, size, size, c) {
        Vector2 dir = target - p;
        if (dir.length() > 0) velocity = dir.normalize() * speed;
        else velocity = { speed, 0 };
    }

    void update() override {
        pos += velocity;
        if (pos.x < 0 || pos.x > WIDTH || pos.y < 0 || pos.y > HEIGHT) die();
    }
};

class Player : public Entity {
private:
    Vector2 velocity;
    int health;
    int shoot_timeout;
    float speed;

public:
    Player() 
        : Entity({ WIDTH / 2, HEIGHT / 2 }, 30, 30, { 255, 255, 255, 255 }), 
          health(100), shoot_timeout(0), speed(4.0f) {}

    void update() override {
        if (shoot_timeout > 0) shoot_timeout--;
    }

    void move(Vector2 move_vector) {
        if (move_vector.length() > 0) {
            velocity = move_vector.normalize() * speed;
            pos += velocity;
            clamp_position();
        } else {
            velocity = { 0, 0 };
        }
    }

    void clamp_position() {
        if (pos.x < rect.w / 2) pos.x = rect.w / 2;
        if (pos.x > WIDTH - rect.w / 2) pos.x = WIDTH - rect.w / 2;
        if (pos.y < rect.h / 2) pos.y = rect.h / 2;
        if (pos.y > HEIGHT - rect.h / 2) pos.y = HEIGHT - rect.h / 2;
    }

    bool can_shoot() const { return shoot_timeout == 0; }
    void reset_shoot_timeout() { shoot_timeout = 10; }
    void take_damage(int amount) { health -= amount; }
    int get_health() const { return health; }
    Vector2 get_velocity() const { return velocity; }
};

class Enemy : public Entity {
private:
    Player* player_ptr;
    float dist;
    float angle;
    float spiral_speed;
    float approach_speed;
    float min_dist;
    int health;
    int shoot_timeout;
    std::vector<Bullet*>& bullets_ref;

public:
    Enemy(Player* p, std::vector<Bullet*>& bullets) 
        : Entity({0, 0}, 25, 25, { 255, 50, 50, 255 }), 
          player_ptr(p), 
          approach_speed(1.0f), min_dist(150.0f), 
          health(100), 
          bullets_ref(bullets) {
        
        spawn_at_edge();
        Vector2 rel = pos - player_ptr->get_pos();
        dist = rel.length();
        angle = std::atan2(rel.y, rel.x);
        spiral_speed = (Random::get_float(1, 5) * (Random::get_int(0, 1) ? 1 : -1)) / 100.0f;
        shoot_timeout = Random::get_int(60, 120);
    }

    void spawn_at_edge() {
        int side = Random::get_int(0, 3);
        if (side == 0) pos = { Random::get_float(0, WIDTH), -30 };
        else if (side == 1) pos = { WIDTH + 30, Random::get_float(0, HEIGHT) };
        else if (side == 2) pos = { Random::get_float(0, WIDTH), HEIGHT + 30 };
        else pos = { -30, Random::get_float(0, HEIGHT) };
    }

    void update() override {
        angle += spiral_speed;
        if (dist > min_dist) dist -= approach_speed;
        Vector2 offset = { std::cos(angle) * dist, std::sin(angle) * dist };
        pos = player_ptr->get_pos() + offset;

        if (--shoot_timeout <= 0) {
            shoot();
            shoot_timeout = Random::get_int(90, 180);
        }
    }

    void shoot() {
        float b_speed = 5.0f;
        float dist_to_p = pos.distance_to(player_ptr->get_pos());
        float time_to_reach = dist_to_p / b_speed;
        Vector2 predicted_pos = player_ptr->get_pos() + (player_ptr->get_velocity() * time_to_reach);
        bullets_ref.push_back(new Bullet(pos, predicted_pos, { 255, 100, 100, 255 }, 8, b_speed));
    }

    void take_damage(int amount) {
        health -= amount;
        if (health <= 0) die();
    }
};

class Background {
private:
    SDL_Texture* texture;
    int width, height;
    float moved;

public:
    Background(SDL_Renderer* renderer) : texture(nullptr), width(0), height(0), moved(0) {
        texture = IMG_LoadTexture(renderer, "back.jpg");
        if (texture) SDL_QueryTexture(texture, NULL, NULL, &width, &height);
    }

    ~Background() {
        if (texture) SDL_DestroyTexture(texture);
    }

    void update() {
        moved += 1.0f;
        if (moved >= width) moved = 0;
    }

    void draw(SDL_Renderer* renderer, int ox, int oy) {
        if (!texture) return;
        SDL_Rect r1 = { (int)(-moved) + ox, oy, width, HEIGHT };
        SDL_Rect r2 = { (int)(width - moved) + ox, oy, width, HEIGHT };
        SDL_RenderCopy(renderer, texture, NULL, &r1);
        SDL_RenderCopy(renderer, texture, NULL, &r2);
    }
};

class Model {
public:
    int input_size;
    int hidden_size;
    int output_size;
    
    std::vector<std::vector<float>> w1; // input to hidden
    std::vector<std::vector<float>> w2; // hidden to output
    float lr = 0.01f;

    struct Experience {
        std::vector<float> state;
        int action;
        float reward;
        std::vector<float> probs;
    };
    std::vector<Experience> memory;

    Model(int in, int out) : input_size(in), hidden_size(16), output_size(out) {
        w1.resize(hidden_size, std::vector<float>(input_size));
        w2.resize(output_size, std::vector<float>(hidden_size));
        
        for (auto& row : w1) for (auto& w : row) w = Random::get_float(-0.5, 0.5);
        for (auto& row : w2) for (auto& w : row) w = Random::get_float(-0.5, 0.5);
    }

    std::vector<float> softmax(const std::vector<float>& x) {
        std::vector<float> out(x.size());
        float maxv = *std::max_element(x.begin(), x.end());
        float sum = 0;
        for (size_t i = 0; i < x.size(); i++) {
            out[i] = std::exp(x[i] - maxv);
            sum += out[i];
        }
        for (float& v : out) v /= sum;
        return out;
    }

    std::vector<float> forward_probs(const std::vector<float>& state) {
        std::vector<float> h(hidden_size);
        // hidden layer
        for (int i = 0; i < hidden_size; i++) {
            float sum = 0;
            for (int j = 0; j < input_size; j++)
                sum += w1[i][j] * state[j];
            h[i] = std::tanh(sum);
        }

        // output layer
        std::vector<float> logits(output_size);
        for (int i = 0; i < output_size; i++) {
            float sum = 0;
            for (int j = 0; j < hidden_size; j++)
                sum += w2[i][j] * h[j];
            logits[i] = sum;
        }
        return softmax(logits);
    }

    int sample_action(const std::vector<float>& probs) {
        float r = Random::get_float(0, 1);
        float cum = 0;
        for (int i = 0; i < (int)probs.size(); i++) {
            cum += probs[i];
            if (r < cum) return i;
        }
        return (int)probs.size() - 1;
    }

    void remember(std::vector<float> s, int a, float r, std::vector<float> p) {
        memory.push_back({s, a, r, p});
    }

    void train() {
        if (memory.empty()) return;
        
        float gamma = 0.99f;
        float G = 0;
        float total_r = 0;
        for (auto& m : memory) total_r += m.reward;
        std::cout << "Training Policy Gradient... Avg Reward/Step: " << (total_r / (float)memory.size()) << std::endl;

        // Discounted return iteration (backwards)
        for (int t = (int)memory.size() - 1; t >= 0; t--) {
            G = memory[t].reward + gamma * G;
            auto& m = memory[t];

            // Re-calculate hidden state for gradients
            std::vector<float> h(hidden_size);
            for (int i = 0; i < hidden_size; i++) {
                float sum = 0;
                for (int j = 0; j < input_size; j++)
                    sum += w1[i][j] * m.state[j];
                h[i] = std::tanh(sum);
            }

            // Update output layer w2
            for (int i = 0; i < output_size; i++) {
                // Policy gradient: (target_action - probability) * Return
                float grad = ((i == m.action ? 1.0f : 0.0f) - m.probs[i]);
                for (int j = 0; j < hidden_size; j++) {
                    w2[i][j] += lr * G * grad * h[j];
                    // Basic clipping
                    if (w2[i][j] > 5.0f) w2[i][j] = 5.0f;
                    if (w2[i][j] < -5.0f) w2[i][j] = -5.0f;
                }
            }
            
            // Note: Simple REINFORCE often skips w1 backprop for speed in these games
            // but we could add it if needed.
        }
        memory.clear();
    }
};

// Main Game Controller
class Game {
private:
    SDL_Window* window;
    SDL_Renderer* renderer;
    TTF_Font* font;
    
    Player* player;
    Model* model;
    Background* bg;
    
    std::vector<Enemy*> enemies;
    std::vector<Bullet*> player_bullets;
    std::vector<Bullet*> enemy_bullets;
    std::vector<Particle*> particles;

    Vector2 prev_enemy_pos;
    int score;
    float total_reward;
    int spawn_timer;
    int shake_amount;
    bool running;

public:
    Game() : window(nullptr), renderer(nullptr), font(nullptr), 
             player(nullptr), model(nullptr), bg(nullptr),
             score(0), total_reward(0), spawn_timer(0), shake_amount(0), running(false) {}

    ~Game() { cleanup(); }

    bool init() {
        if (SDL_Init(SDL_INIT_VIDEO) < 0) return false;
        if (!(IMG_Init(IMG_INIT_JPG) & IMG_INIT_JPG)) return false;
        if (TTF_Init() < 0) return false;

        window = SDL_CreateWindow("Spiral Shooter OOP", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WIDTH, HEIGHT, 0);
        if (!window) return false;

        renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
        if (!renderer) return false;

        font = TTF_OpenFont("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24);
        if (!font) font = TTF_OpenFont("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", 24);

        player = new Player();
        model = new Model(9, 6); // 8 inputs + 1 bias, 6 outputs
        bg = new Background(renderer);
        running = true;
        return true;
    }

    void run() {
        while (running) {
            handle_events();
            update();
            render();
            SDL_Delay(16);
        }
    }

private:
    void handle_events() {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) running = false;
        }
    }

    std::vector<float> get_state() {
        std::vector<float> s;
        s.push_back(player->get_pos().x / WIDTH);
        s.push_back(player->get_pos().y / HEIGHT);
        s.push_back(player->get_velocity().x / 10.0f);
        s.push_back(player->get_velocity().y / 10.0f);

        float min_d = 1e9;
        Vector2 nearest_rel = {0,0};
        Vector2 nearest_pos = {0,0};
        bool found_enemy = false;

        for (auto e : enemies) {
            float d = player->get_pos().distance_to(e->get_pos());
            if (d < min_d) {
                min_d = d;
                nearest_rel = e->get_pos() - player->get_pos();
                nearest_pos = e->get_pos();
                found_enemy = true;
            }
        }

        s.push_back(nearest_rel.x / WIDTH);
        s.push_back(nearest_rel.y / HEIGHT);

        // enemy velocity (prediction)
        Vector2 enemy_vel = {0, 0};
        if (found_enemy) {
            if (prev_enemy_pos.length() > 0) { 
                enemy_vel = nearest_pos - prev_enemy_pos;
                // Clamp velocity to prevent state spikes when nearest enemy switches
                if (enemy_vel.length() > 50.0f) enemy_vel = {0, 0}; 
            }
            prev_enemy_pos = nearest_pos;
        } else {
            prev_enemy_pos = {0, 0};
        }
        s.push_back(enemy_vel.x / 10.0f);
        s.push_back(enemy_vel.y / 10.0f);
        
        // Add BIAS term (constant 1.0) so model can learn baseline action preference
        s.push_back(1.0f); 

        return s;
    }

    void update() {
        std::vector<float> state = get_state();
        auto probs = model->forward_probs(state);
        int action = model->sample_action(probs);

        // Movement
        Vector2 move_dir = {0, 0};
        if (action == UP) move_dir.y -= 1;
        if (action == DOWN) move_dir.y += 1;
        if (action == LEFT) move_dir.x -= 1;
        if (action == RIGHT) move_dir.x += 1;
        player->move(move_dir);
        player->update();

        // Shooting
        Vector2 nearest_enemy_pos = player->get_pos();
        float nearest_enemy_distance = 1e9;
        bool found_enemy = false;
        for (auto en : enemies) {
            float d = player->get_pos().distance_to(en->get_pos());
            if (d < nearest_enemy_distance) {
                nearest_enemy_distance = d;
                nearest_enemy_pos = en->get_pos();
                found_enemy = true;
            }
        }

        // Rewards
        float reward = 0;

        // survival
        reward += 0.01f;

        // high reward for close + shooting
        if (action == SHOOT && player->can_shoot()) {
            player_bullets.push_back(new Bullet(player->get_pos(), nearest_enemy_pos, { 130, 200, 77, 255 }, 10, 10));
            player->reset_shoot_timeout();
            if (found_enemy) {
                reward += (200.0f - nearest_enemy_distance) * 0.05f;
            }
        }

        // but also risk
        if (found_enemy && nearest_enemy_distance < 80) {
            reward -= 0.1f;
        }

        // reward aiming (alignment)
        if (found_enemy) {
            Vector2 to_enemy = nearest_enemy_pos - player->get_pos();
            Vector2 forward = player->get_velocity();
            if (forward.length() > 0) {
                float align = (to_enemy.normalize().x * forward.normalize().x +
                               to_enemy.normalize().y * forward.normalize().y);
                reward += align * 0.02f;
            }
        }

        // Spawning
        if (++spawn_timer >= std::max(30, 90 - (score / 500)) && enemies.size() < 5) {
            enemies.push_back(new Enemy(player, enemy_bullets));
            spawn_timer = 0;
        }

        bg->update();
        update_entities(reward);
        
        total_reward += reward;
        model->remember(state, action, reward, probs);
    }

    void update_entities(float& reward) {
        // Enemies
        for (auto it = enemies.begin(); it != enemies.end();) {
            (*it)->update();
            
            bool enemy_dead = false;
            SDL_Rect er = (*it)->get_rect();

            // Collision with player bullets
            for (auto bit = player_bullets.begin(); bit != player_bullets.end();) {
                SDL_Rect br = (*bit)->get_rect();
                if (SDL_HasIntersection(&er, &br)) {
                    // Proximity damage logic:
                    float dist = player->get_pos().distance_to((*it)->get_pos());
                    int damage = 25;
                    
                    if (dist < CUBE_SIZE) {
                        damage = 100; // Point-blank one-shot
                    } else {
                        // Inverse square scaling: Damage = 90 * (CUBE_SIZE / dist)^2
                        float scale = (CUBE_SIZE * CUBE_SIZE) / (dist * dist);
                        damage = (int)(90.0f * scale);
                        if (damage < 50 && dist > 300) damage = 50; 
                        if (damage > 90) damage = 90;
                    }

                    (*it)->take_damage(damage);
                    reward += 15.0f; // hit reward
                    delete *bit;
                    bit = player_bullets.erase(bit);
                    if (!(*it)->is_alive()) {
                        spawn_particles((*it)->get_pos());
                        score += 100;
                        reward += 50.0f; // kill reward
                        enemy_dead = true;
                    }
                } else ++bit;
            }

            if (enemy_dead) {
                delete *it;
                it = enemies.erase(it);
            } else {
                SDL_Rect pr = player->get_rect();
                if (SDL_HasIntersection(&er, &pr)) {
                    reward -= 10.0f; // death penalty
                    std::cout << "GAME OVER: Touched enemy! | Score: " << score << " | Reward: " << total_reward + reward << std::endl;
                    model->train();
                    running = false;
                }
                ++it;
            }
        }

        // Bullets
        auto update_bullets = [&](std::vector<Bullet*>& bullets) {
            for (auto it = bullets.begin(); it != bullets.end();) {
                (*it)->update();
                
                // Optimized near miss reward
                float min_d = 1e9;
                for (auto e : enemies) {
                    float d = (*it)->get_pos().distance_to(e->get_pos());
                    if (d < min_d) min_d = d;
                }
                if (min_d < 50) reward += 0.01f;

                if (!(*it)->is_alive()) { delete *it; it = bullets.erase(it); }
                else ++it;
            }
        };
        update_bullets(player_bullets);

        // Enemy Bullets vs Player
        for (auto it = enemy_bullets.begin(); it != enemy_bullets.end();) {
            (*it)->update();
            SDL_Rect br = (*it)->get_rect();
            SDL_Rect pr = player->get_rect();
            if (SDL_HasIntersection(&br, &pr)) {
                // High-risk proximity damage logic for player
                float dist = player->get_pos().distance_to((*it)->get_pos());
                int damage = 25;

                if (dist < CUBE_SIZE) {
                    damage = 100; // Point-blank instant/lethal
                } else {
                    float scale = (CUBE_SIZE * CUBE_SIZE) / (dist * dist);
                    damage = (int)(90.0f * scale);
                    if (damage < 50 && dist > 300) damage = 50;
                    if (damage > 90) damage = 90;
                }

                player->take_damage(damage);
                reward -= (float)damage; 
                shake_amount = 10;
                delete *it;
                it = enemy_bullets.erase(it);
                if (player->get_health() <= 0) {
                    reward -= 10.0f; // death penalty
                    std::cout << "GAME OVER: HP zero! | Score: " << score << " | Reward: " << total_reward + reward << std::endl;
                    model->train();
                    running = false;
                }
            } else if (!(*it)->is_alive()) { delete *it; it = enemy_bullets.erase(it); }
            else ++it;
        }

        // Particles
        for (auto it = particles.begin(); it != particles.end();) {
            (*it)->update();
            if (!(*it)->is_alive()) { delete *it; it = particles.erase(it); }
            else ++it;
        }
    }

    void spawn_particles(Vector2 pos) {
        for (int i = 0; i < 10; i++) particles.push_back(new Particle(pos, { 255, 50, 50, 255 }));
    }

    void render() {
        int ox = 0, oy = 0;
        if (shake_amount > 0) {
            ox = Random::get_int(-shake_amount, shake_amount);
            oy = Random::get_int(-shake_amount, shake_amount);
            shake_amount--;
        }

        SDL_SetRenderDrawColor(renderer, 30, 30, 30, 255);
        SDL_RenderClear(renderer);

        bg->draw(renderer, ox, oy);
        for (auto p : particles) p->draw(renderer, ox, oy);
        for (auto b : player_bullets) b->draw(renderer, ox, oy);
        for (auto b : enemy_bullets) b->draw(renderer, ox, oy);
        for (auto e : enemies) e->draw(renderer, ox, oy);
        player->draw(renderer, ox, oy);

        draw_ui();

        SDL_RenderPresent(renderer);
    }

    void draw_ui() {
        if (!font) return;
        char buf[64];
        
        auto render_text = [&](const char* text, int x, int y) {
            SDL_Surface* surf = TTF_RenderText_Solid(font, text, { 255, 255, 255, 255 });
            SDL_Texture* tex = SDL_CreateTextureFromSurface(renderer, surf);
            SDL_Rect dest = { x, y, surf->w, surf->h };
            SDL_RenderCopy(renderer, tex, NULL, &dest);
            SDL_FreeSurface(surf);
            SDL_DestroyTexture(tex);
        };

        sprintf(buf, "HP: %d", player->get_health());
        render_text(buf, 10, 10);
        sprintf(buf, "Score: %d", score);
        render_text(buf, 10, 40);
        sprintf(buf, "Reward: %.2f", total_reward);
        render_text(buf, 10, 70);
    }

    void cleanup() {
        delete player; player = nullptr;
        delete model; model = nullptr;
        delete bg; bg = nullptr;
        
        for (auto e : enemies) delete e; 
        enemies.clear();
        
        for (auto b : player_bullets) delete b; 
        player_bullets.clear();
        
        for (auto b : enemy_bullets) delete b; 
        enemy_bullets.clear();
        
        for (auto p : particles) delete p; 
        particles.clear();

        if (font) TTF_CloseFont(font);
        if (renderer) SDL_DestroyRenderer(renderer);
        if (window) SDL_DestroyWindow(window);
        TTF_Quit();
        IMG_Quit();
        SDL_Quit();
    }
};

int main(int argc, char* argv[]) {
    Game game;
    if (game.init()) {
        game.run();
    }
    return 0;
}
