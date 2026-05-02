#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_ttf.h>
#include <iostream>
#include <vector>
#include <deque>
#include <cmath>
#include <algorithm>
#include <random>
#include <fstream>
#include <cstring>
#include <numeric>
#include <cassert>

// ─────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────
const int WIDTH  = 1600;
const int HEIGHT = 900;

// ─────────────────────────────────────────────
// Action indices
// ─────────────────────────────────────────────
enum Action { MOVE_X = 0, MOVE_Y, AIM_X, AIM_Y, SHOOT_PROB, NUM_ACTIONS };

// ─────────────────────────────────────────────
// Math Utilities
// ─────────────────────────────────────────────
struct Vector2 {
    float x = 0, y = 0;
    Vector2(float x = 0, float y = 0) : x(x), y(y) {}
    Vector2 operator+(const Vector2& v) const { return {x+v.x, y+v.y}; }
    Vector2 operator-(const Vector2& v) const { return {x-v.x, y-v.y}; }
    Vector2 operator*(float s)          const { return {x*s,   y*s};   }
    Vector2 operator/(float s)          const { return {x/s,   y/s};   }
    Vector2& operator+=(const Vector2& v){ x+=v.x; y+=v.y; return *this; }
    Vector2& operator-=(const Vector2& v){ x-=v.x; y-=v.y; return *this; }
    float length()   const { return std::sqrt(x*x + y*y); }
    float dot(const Vector2& v) const { return x*v.x + y*v.y; }
    Vector2 normalize() const {
        float l = length();
        return (l > 1e-6f) ? Vector2{x/l, y/l} : Vector2{0,0};
    }
    float distance_to(const Vector2& v) const { return (*this - v).length(); }
};

// ─────────────────────────────────────────────
// RNG helpers
// ─────────────────────────────────────────────
class RNG {
    static std::mt19937& gen() {
        static std::mt19937 g(std::random_device{}());
        return g;
    }
public:
    static float flt(float lo, float hi) {
        return std::uniform_real_distribution<float>(lo, hi)(gen());
    }
    static int rng_int(int lo, int hi) {
        return std::uniform_int_distribution<int>(lo, hi)(gen());
    }
    // Normal distribution sample
    static float normal(float mean = 0.f, float std = 1.f) {
        return std::normal_distribution<float>(mean, std)(gen());
    }
};

// ─────────────────────────────────────────────
// NN helper: linear algebra
// ─────────────────────────────────────────────
using Mat = std::vector<std::vector<float>>;
using Vec = std::vector<float>;

static Mat make_mat(int rows, int cols, float val = 0.f) {
    return Mat(rows, Vec(cols, val));
}

static Vec mat_vec_mul(const Mat& W, const Vec& x) {
    Vec out(W.size(), 0.f);
    for (int i = 0; i < (int)W.size(); i++)
        for (int j = 0; j < (int)x.size(); j++)
            out[i] += W[i][j] * x[j];
    return out;
}

static void add_bias(Vec& v, const Vec& b) {
    for (int i = 0; i < (int)v.size(); i++) v[i] += b[i];
}

static Vec leaky_relu(const Vec& v, float alpha = 0.01f) {
    Vec out(v.size());
    for (int i = 0; i < (int)v.size(); i++)
        out[i] = v[i] >= 0.f ? v[i] : alpha * v[i];
    return out;
}

static Vec d_leaky_relu(const Vec& pre, float alpha = 0.01f) {
    Vec out(pre.size());
    for (int i = 0; i < (int)pre.size(); i++)
        out[i] = pre[i] >= 0.f ? 1.f : alpha;
    return out;
}

// Layer Normalization (zero mean, unit variance per sample)
static void layer_norm(Vec& v, float eps = 1e-5f) {
    float mean = 0, var = 0;
    for (float x : v) mean += x;
    mean /= v.size();
    for (float x : v) var += (x - mean) * (x - mean);
    var /= v.size();
    float inv = 1.f / std::sqrt(var + eps);
    for (float& x : v) x = (x - mean) * inv;
}

// ─────────────────────────────────────────────
// Adam optimizer state
// ─────────────────────────────────────────────
struct AdamState {
    Mat m, v;   // first/second moment for weight matrix
    Vec mb, vb; // for bias vector
    int t = 0;

    AdamState() = default;
    AdamState(int rows, int cols)
        : m(make_mat(rows, cols)), v(make_mat(rows, cols)),
          mb(rows, 0.f), vb(rows, 0.f) {}
};

static void adam_update(Mat& W, Vec& b,
                        const Mat& dW, const Vec& db,
                        AdamState& st,
                        float lr, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f)
{
    st.t++;
    float bc1 = 1.f - std::pow(beta1, st.t);
    float bc2 = 1.f - std::pow(beta2, st.t);

    for (int i = 0; i < (int)W.size(); i++) {
        for (int j = 0; j < (int)W[0].size(); j++) {
            st.m[i][j] = beta1 * st.m[i][j] + (1.f-beta1) * dW[i][j];
            st.v[i][j] = beta2 * st.v[i][j] + (1.f-beta2) * dW[i][j] * dW[i][j];
            float mhat = st.m[i][j] / bc1;
            float vhat = st.v[i][j] / bc2;
            W[i][j] += lr * mhat / (std::sqrt(vhat) + eps);
        }
        st.mb[i] = beta1 * st.mb[i] + (1.f-beta1) * db[i];
        st.vb[i] = beta2 * st.vb[i] + (1.f-beta2) * db[i] * db[i];
        float mhat = st.mb[i] / bc1;
        float vhat = st.vb[i] / bc2;
        b[i] += lr * mhat / (std::sqrt(vhat) + eps);
    }
}

// ─────────────────────────────────────────────
// Policy Network  (3-hidden-layer MLP)
//   Input  : STATE_DIM
//   Hidden : 128 → 128 → 64  (LeakyReLU + LayerNorm)
//   Output : NUM_ACTIONS      (tanh for 0-3, sigmoid for 4)
// ─────────────────────────────────────────────
static constexpr int STATE_DIM  = 20; // expanded state
static constexpr int H1 = 128, H2 = 128, H3 = 64;

struct PolicyNet {
    // Weights
    Mat w1, w2, w3, w4;
    Vec b1, b2, b3, b4;
    // Adam states
    AdamState as1, as2, as3, as4;
    float lr = 3e-4f;

    PolicyNet() {
        auto he = [](int fan_in, int fan_out) {
            float s = std::sqrt(2.f / fan_in);
            return RNG::normal(0, s);
        };

        auto init_mat = [&](Mat& W, Vec& b, int rows, int cols) {
            W = make_mat(rows, cols);
            b.assign(rows, 0.f);
            for (auto& row : W) for (auto& w : row) w = he(cols, rows);
        };

        init_mat(w1, b1, H1, STATE_DIM);
        init_mat(w2, b2, H2, H1);
        init_mat(w3, b3, H3, H2);
        init_mat(w4, b4, NUM_ACTIONS, H3);

        // shoot bias: start near 0 to avoid random shooting
        b4[SHOOT_PROB] = -1.f;

        as1 = AdamState(H1, STATE_DIM);
        as2 = AdamState(H2, H1);
        as3 = AdamState(H3, H2);
        as4 = AdamState(NUM_ACTIONS, H3);
    }

    struct Cache { Vec h1, h2, h3, pre1, pre2, pre3; };

    Vec forward(const Vec& s, Cache* cache = nullptr, bool training = false) const {
        // Layer 1
        Vec pre1 = mat_vec_mul(w1, s); add_bias(pre1, b1);
        Vec h1   = leaky_relu(pre1);   layer_norm(h1);

        // Dropout (training only) — 20%
        if (training) for (auto& v : h1) if (RNG::flt(0,1) < 0.15f) v = 0.f;

        // Layer 2
        Vec pre2 = mat_vec_mul(w2, h1); add_bias(pre2, b2);
        Vec h2   = leaky_relu(pre2);    layer_norm(h2);
        if (training) for (auto& v : h2) if (RNG::flt(0,1) < 0.15f) v = 0.f;

        // Layer 3
        Vec pre3 = mat_vec_mul(w3, h2); add_bias(pre3, b3);
        Vec h3   = leaky_relu(pre3);    layer_norm(h3);

        // Output
        Vec out = mat_vec_mul(w4, h3); add_bias(out, b4);

        // Activations
        for (int i = 0; i < 4; i++) out[i] = std::tanh(out[i]);
        out[SHOOT_PROB] = 1.f / (1.f + std::exp(-out[SHOOT_PROB]));

        if (cache) *cache = {h1, h2, h3, pre1, pre2, pre3};
        return out;
    }

    // Full forward for training (returns activations without dropout applied to final pass)
    Vec forward_train(const Vec& s, Cache& cache) {
        Vec pre1 = mat_vec_mul(w1, s); add_bias(pre1, b1);
        Vec h1   = leaky_relu(pre1);   layer_norm(h1);
        Vec pre2 = mat_vec_mul(w2, h1); add_bias(pre2, b2);
        Vec h2   = leaky_relu(pre2);    layer_norm(h2);
        Vec pre3 = mat_vec_mul(w3, h2); add_bias(pre3, b3);
        Vec h3   = leaky_relu(pre3);    layer_norm(h3);
        Vec out  = mat_vec_mul(w4, h3); add_bias(out, b4);
        for (int i = 0; i < 4; i++) out[i] = std::tanh(out[i]);
        out[SHOOT_PROB] = 1.f / (1.f + std::exp(-out[SHOOT_PROB]));
        cache = {h1, h2, h3, pre1, pre2, pre3};
        return out;
    }

    void train_batch(const std::vector<Vec>& states,
                     const std::vector<Vec>& outputs_taken,
                     const std::vector<float>& returns)
    {
        int N = (int)states.size();

        // Normalize returns
        float mean_r = 0, std_r = 0;
        for (float r : returns) mean_r += r;
        mean_r /= N;
        for (float r : returns) std_r += (r - mean_r) * (r - mean_r);
        std_r = std::sqrt(std_r / N + 1e-8f);

        // Accumulate gradients
        Mat dw1 = make_mat(H1, STATE_DIM), dw2 = make_mat(H2, H1),
            dw3 = make_mat(H3, H2),       dw4 = make_mat(NUM_ACTIONS, H3);
        Vec db1(H1,0), db2(H2,0), db3(H3,0), db4(NUM_ACTIONS,0);

        for (int idx = 0; idx < N; idx++) {
            float G = (returns[idx] - mean_r) / std_r;

            Cache cache;
            Vec pred = forward_train(states[idx], cache);

            // Output error: REINFORCE gradient
            Vec out_err(NUM_ACTIONS, 0.f);
            for (int i = 0; i < 4; i++) {
                // tanh output: gradient of log-prob approximation
                float act = outputs_taken[idx][i];
                float p   = pred[i];
                float dt  = 1.f - p * p; // d_tanh
                out_err[i] = G * act * dt;
            }
            // Sigmoid shoot: binary cross-entropy style
            {
                float act = outputs_taken[idx][SHOOT_PROB];
                float p   = pred[SHOOT_PROB];
                out_err[SHOOT_PROB] = G * (act - p) * p * (1.f - p);
            }

            // Backprop through layer 4
            Vec h3_err(H3, 0.f);
            for (int i = 0; i < H3; i++) {
                float e = 0;
                for (int j = 0; j < NUM_ACTIONS; j++) {
                    dw4[j][i] += out_err[j] * cache.h3[i];
                    e         += w4[j][i] * out_err[j];
                }
                h3_err[i] = e;
            }
            for (int j = 0; j < NUM_ACTIONS; j++) db4[j] += out_err[j];

            // Layer 3
            Vec d3  = d_leaky_relu(cache.pre3);
            Vec h2_err(H2, 0.f);
            for (int i = 0; i < H2; i++) {
                float e = 0;
                for (int j = 0; j < H3; j++) {
                    float g = h3_err[j] * d3[j];
                    dw3[j][i] += g * cache.h2[i];
                    e         += w3[j][i] * g;
                }
                h2_err[i] = e;
            }
            for (int j = 0; j < H3; j++) db3[j] += h3_err[j] * d3[j];

            // Layer 2
            Vec d2  = d_leaky_relu(cache.pre2);
            Vec h1_err(H1, 0.f);
            for (int i = 0; i < H1; i++) {
                float e = 0;
                for (int j = 0; j < H2; j++) {
                    float g = h2_err[j] * d2[j];
                    dw2[j][i] += g * cache.h1[i];
                    e         += w2[j][i] * g;
                }
                h1_err[i] = e;
            }
            for (int j = 0; j < H2; j++) db2[j] += h2_err[j] * d2[j];

            // Layer 1
            Vec d1  = d_leaky_relu(cache.pre1);
            for (int i = 0; i < STATE_DIM; i++) {
                for (int j = 0; j < H1; j++) {
                    dw1[j][i] += h1_err[j] * d1[j] * states[idx][i];
                }
            }
            for (int j = 0; j < H1; j++) db1[j] += h1_err[j] * d1[j];
        }

        float scale = 1.f / N;
        for (auto& row : dw1) for (auto& v : row) v *= scale;
        for (auto& row : dw2) for (auto& v : row) v *= scale;
        for (auto& row : dw3) for (auto& v : row) v *= scale;
        for (auto& row : dw4) for (auto& v : row) v *= scale;
        for (auto& v : db1) v *= scale;
        for (auto& v : db2) v *= scale;
        for (auto& v : db3) v *= scale;
        for (auto& v : db4) v *= scale;

        adam_update(w1, b1, dw1, db1, as1, lr);
        adam_update(w2, b2, dw2, db2, as2, lr);
        adam_update(w3, b3, dw3, db3, as3, lr);
        adam_update(w4, b4, dw4, db4, as4, lr);
    }

    void save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) { std::cerr << "Save failed: " << path << "\n"; return; }
        auto wmat = [&](const Mat& M) {
            for (auto& r : M) f.write((char*)r.data(), r.size()*sizeof(float));
        };
        auto wvec = [&](const Vec& v) {
            f.write((char*)v.data(), v.size()*sizeof(float));
        };
        wmat(w1); wmat(w2); wmat(w3); wmat(w4);
        wvec(b1); wvec(b2); wvec(b3); wvec(b4);
        std::cout << "[Model] Saved to " << path << "\n";
    }

    void load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) { std::cout << "[Model] No weights found, starting fresh.\n"; return; }
        auto rmat = [&](Mat& M) {
            for (auto& r : M) f.read((char*)r.data(), r.size()*sizeof(float));
        };
        auto rvec = [&](Vec& v) {
            f.read((char*)v.data(), v.size()*sizeof(float));
        };
        rmat(w1); rmat(w2); rmat(w3); rmat(w4);
        rvec(b1); rvec(b2); rvec(b3); rvec(b4);
        std::cout << "[Model] Loaded from " << path << "\n";
    }
};

// ─────────────────────────────────────────────
// Experience Replay Buffer
// ─────────────────────────────────────────────
struct Experience {
    Vec state, action;
    float reward;
};

class ReplayBuffer {
    std::deque<Experience> buf;
    int capacity;
public:
    explicit ReplayBuffer(int cap = 8192) : capacity(cap) {}

    void push(Vec s, Vec a, float r) {
        if ((int)buf.size() >= capacity) buf.pop_front();
        buf.push_back({std::move(s), std::move(a), r});
    }

    // Sample a random batch; returns discounted returns (Monte-Carlo style)
    bool sample(int batch, std::vector<Vec>& states, std::vector<Vec>& actions, std::vector<float>& returns) const {
        if ((int)buf.size() < batch) return false;
        std::vector<int> idxs(buf.size());
        std::iota(idxs.begin(), idxs.end(), 0);
        // Random shuffle then take first `batch`
        for (int i = (int)idxs.size()-1; i > 0; i--) {
            int j = RNG::rng_int(0, i);
            std::swap(idxs[i], idxs[j]);
        }
        idxs.resize(batch);

        states.clear(); actions.clear(); returns.clear();
        for (int i : idxs) {
            states.push_back(buf[i].state);
            actions.push_back(buf[i].action);
            returns.push_back(buf[i].reward);
        }
        return true;
    }

    int size() const { return (int)buf.size(); }
};

// ─────────────────────────────────────────────
// Game Entities
// ─────────────────────────────────────────────
class Entity {
protected:
    Vector2 pos;
    int w, h;
    SDL_Color color;
    bool alive = true;
    SDL_Texture* tex = nullptr;
    float angle = 0;
public:
    Entity(Vector2 p, int w, int h, SDL_Color c) : pos(p), w(w), h(h), color(c) {}
    virtual ~Entity() = default;
    virtual void update() = 0;
    virtual void draw(SDL_Renderer* r, int ox, int oy) {
        SDL_Rect rc = { (int)std::round(pos.x - w/2) + ox,
                        (int)std::round(pos.y - h/2) + oy, w, h };
        if (tex) {
            SDL_RenderCopyEx(r, tex, NULL, &rc, (double)angle, NULL, SDL_FLIP_NONE);
        } else {
            // Glow effect
            SDL_SetRenderDrawBlendMode(r, SDL_BLENDMODE_BLEND);
            SDL_SetRenderDrawColor(r, color.r, color.g, color.b, 60);
            SDL_Rect glow = { rc.x - 2, rc.y - 2, w + 4, h + 4 };
            SDL_RenderFillRect(r, &glow);

            // Main body
            SDL_SetRenderDrawColor(r, color.r, color.g, color.b, color.a);
            SDL_RenderFillRect(r, &rc);
        }

        // Hitbox outline (Yellow)
        SDL_SetRenderDrawColor(r, 255, 255, 0, 255);
        SDL_RenderDrawRect(r, &rc);
    }
    bool is_alive() const { return alive; }
    void die()           { alive = false; }
    Vector2 get_pos()    const { return pos; }
    SDL_Rect get_rect()  const { return { (int)(pos.x-w/2), (int)(pos.y-h/2), w, h }; }
    void set_texture(SDL_Texture* t) { tex = t; }
    void set_angle(float a) { angle = a; }
};

class Particle : public Entity {
    Vector2 vel;
    int life;
public:
    Particle(Vector2 p, SDL_Color c)
        : Entity(p, 4, 4, c), vel(RNG::flt(-3,3), RNG::flt(-3,3)), life(30) {}
    void update() override { pos += vel; if (--life <= 0) die(); }
};

class Bullet : public Entity {
    Vector2 vel;
public:
    Bullet(Vector2 p, Vector2 target, SDL_Color c, int sz, float spd)
        : Entity(p, sz, sz, c) {
        vel = (target - p).normalize() * spd;
    }
    void update() override {
        pos += vel;
        if (pos.x < -50 || pos.x > WIDTH+50 || pos.y < -50 || pos.y > HEIGHT+50) die();
    }
    Vector2 get_vel() const { return vel; }
};

class Player : public Entity {
    Vector2 velocity;
    int health = 100;
    int shoot_cd = 0;
    float speed = 6.f;
public:
    Player() : Entity({WIDTH/2.f, HEIGHT/2.f}, 30, 30, {200, 220, 255, 255}) {}
    void update() override { if (shoot_cd > 0) shoot_cd--; }
    void move(Vector2 dir, bool restricted = false) {
        if (dir.length() > 0) {
            if (dir.length() > 1.f) dir = dir.normalize();
            velocity = dir * speed;
            pos += velocity;
        } else { velocity = {0,0}; }
        if (restricted) {
            int cx = WIDTH/2, cy = HEIGHT/2;
            clamp(cx-300, cx+300, cy-300, cy+300);
        } else clamp(0, WIDTH, 0, HEIGHT);
    }
    void clamp(int x0, int x1, int y0, int y1) {
        if (pos.x < x0+w/2) pos.x = x0+w/2;
        if (pos.x > x1-w/2) pos.x = x1-w/2;
        if (pos.y < y0+h/2) pos.y = y0+h/2;
        if (pos.y > y1-h/2) pos.y = y1-h/2;
    }
    bool can_shoot()  const { return shoot_cd == 0; }
    void fire()             { shoot_cd = 8; }
    void take_damage(int d) { health -= d; }
    int  get_health() const { return health; }
    int  get_cd()     const { return shoot_cd; }
    Vector2 get_vel() const { return velocity; }
};

class Enemy : public Entity {
    Player*              player_ptr;
    std::vector<Bullet*>& bullets_ref;
    float  spiral_speed;
    float  min_dist;
    int    health = 150;
    int    shoot_cd;
    // For velocity tracking by AI
    Vector2 prev_pos;
    Vector2 vel_smooth;
public:
    Enemy(Player* p, std::vector<Bullet*>& eb)
        : Entity({0,0}, 30, 30, {255, 80, 80, 255}),
          player_ptr(p), bullets_ref(eb), min_dist(160.f)
    {
        // Spawn at edge
        int side = RNG::rng_int(0,3);
        if      (side==0) pos = {RNG::flt(0,WIDTH), -35};
        else if (side==1) pos = {(float)WIDTH+35, RNG::flt(0,HEIGHT)};
        else if (side==2) pos = {RNG::flt(0,WIDTH), (float)HEIGHT+35};
        else              pos = {-35, RNG::flt(0,HEIGHT)};

        spiral_speed = RNG::flt(1.5f, 4.5f) * (RNG::rng_int(0,1) ? 1 : -1) / 100.f;
        shoot_cd     = RNG::rng_int(120, 240);
        prev_pos     = pos;
        vel_smooth   = {0,0};
    }

    void update() override {
        Vector2 to_p   = player_ptr->get_pos() - pos;
        float   dist   = to_p.length();
        Vector2 dir    = to_p.normalize();

        // Update angle to face player
        angle = std::atan2(to_p.y, to_p.x) * 180.0f / 3.14159265f;

        if (dist > min_dist)           pos += dir * 2.2f;
        else if (dist < min_dist-20.f) pos -= dir * 1.1f;

        Vector2 perp = {-dir.y, dir.x};
        pos += perp * (spiral_speed * 140.f);

        // Track own velocity for AI exposure
        Vector2 raw  = pos - prev_pos;
        vel_smooth   = vel_smooth * 0.7f + raw * 0.3f;
        prev_pos     = pos;

        if (--shoot_cd <= 0) {
            float bspd = 5.2f;
            float t    = dist / bspd;
            Vector2 pred = player_ptr->get_pos() + player_ptr->get_vel() * t;
            bullets_ref.push_back(new Bullet(pos, pred, {255, 140, 80, 255}, 8, bspd));
            shoot_cd = RNG::rng_int(80, 160);
        }
    }

    void take_damage(int d) { health -= d; if (health <= 0) die(); }
    int     get_health()    const { return health; }
    Vector2 get_vel_smooth()const { return vel_smooth; }
};

class Background {
    SDL_Texture* tex = nullptr;
public:
    Background(SDL_Renderer* r) {
        tex = IMG_LoadTexture(r, "back.png");
        if (tex) {
            SDL_SetTextureColorMod(tex, 150, 150, 200);
        }
    }
    ~Background() { if (tex) SDL_DestroyTexture(tex); }
    void update() {}
    void draw(SDL_Renderer* r, int ox, int oy) {
        if (!tex) return;
        SDL_Rect dst = {ox, oy, WIDTH, HEIGHT};
        SDL_RenderCopy(r, tex, NULL, &dst);
    }
};

// ─────────────────────────────────────────────
// Game
// ─────────────────────────────────────────────
class Game {
    SDL_Window*   window   = nullptr;
    SDL_Renderer* renderer = nullptr;
    TTF_Font*     font     = nullptr;

    Player*     player = nullptr;
    PolicyNet*  net    = nullptr;
    Background* bg     = nullptr;

    SDL_Texture* player_tex = nullptr;
    SDL_Texture* enemy_tex  = nullptr;

    std::vector<Enemy*>    enemies;
    std::vector<Bullet*>   p_bullets, e_bullets;
    std::vector<Particle*> particles;

    ReplayBuffer replay{8192};

    int   score        = 0;
    float total_reward = 0.f;
    int   spawn_timer  = 0;
    int   shake        = 0;
    bool  running      = true;
    bool  ai_mode      = true;

    // Epsilon for exploration
    float epsilon = 0.25f;
    static constexpr float EPS_MIN  = 0.04f;
    static constexpr float EPS_DECAY= 0.999f;

    int   train_every = 64;  // steps between training
    int   step_count  = 0;
    int   edge_timer  = 0;   // track edge camping

    // Nearest-bullet distance last frame (for dodge reward)
    float prev_min_b_dist = 1e9f;

    // Enemy velocity smoothing (for state)
    Vector2 tracked_enemy_vel = {0,0};
    Vector2 tracked_enemy_pos = {0,0};

    uint8_t keys[SDL_NUM_SCANCODES] = {};

    // Hit/accuracy stats
    int shots_fired = 0, shots_hit = 0;

public:
    ~Game() { cleanup(); }

    bool init(bool ai) {
        ai_mode = ai;
        if (SDL_Init(SDL_INIT_VIDEO) < 0) { std::cerr << SDL_GetError() << "\n"; return false; }
        if (!(IMG_Init(IMG_INIT_JPG | IMG_INIT_PNG) & (IMG_INIT_JPG | IMG_INIT_PNG))) {}
        if (TTF_Init() < 0) {}

        window   = SDL_CreateWindow("Spiral Shooter ── AI Enhanced",
                                    SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                    WIDTH, HEIGHT, 0);
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

        player_tex = IMG_LoadTexture(renderer, "Player.png");
        enemy_tex  = IMG_LoadTexture(renderer, "Enemy.png");

        player = new Player();
        if (player_tex) player->set_texture(player_tex);

        net    = new PolicyNet();
        if (ai_mode) net->load("model.weights");
        bg = new Background(renderer);
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
    bool is_visible(Vector2 p) const {
        return (p.x >= 0 && p.x <= WIDTH && p.y >= 0 && p.y <= HEIGHT);
    }

    // ── State Vector (16 features) ──────────────
    Vec get_state() const {
        Vec s;
        Vector2 p_pos = player->get_pos();

        // Player position (normalised)
        s.push_back(p_pos.x / WIDTH);
        s.push_back(p_pos.y / HEIGHT);
        // Player velocity
        s.push_back(player->get_vel().x / 8.f);
        s.push_back(player->get_vel().y / 8.f);

        // Nearest visible enemy: relative direction + distance + velocity
        float  min_ed = 1e9f;
        Vector2 near_e = {0,0};
        Enemy*  near_ep = nullptr;
        bool found_enemy = false;

        for (auto* e : enemies) {
            if (!is_visible(e->get_pos())) continue;
            float d = p_pos.distance_to(e->get_pos());
            if (d < min_ed) { min_ed = d; near_e = e->get_pos(); near_ep = e; found_enemy = true; }
        }

        if (found_enemy) {
            Vector2 e_rel_dir = (near_e - p_pos).normalize();
            s.push_back(e_rel_dir.x);
            s.push_back(e_rel_dir.y);
            s.push_back(std::min(1.f, min_ed / 900.f));   // normalised distance
            // Enemy velocity (for predictive aim)
            Vector2 ev = near_ep->get_vel_smooth();
            s.push_back(ev.x / 5.f);
            s.push_back(ev.y / 5.f);
        } else {
            s.push_back(0.f); s.push_back(0.f); // dir
            s.push_back(1.f);                   // max dist
            s.push_back(0.f); s.push_back(0.f); // vel
        }

        // Nearest visible enemy bullet: direction + distance
        float  min_bd = 1e9f;
        Vector2 near_b = {0,0};
        Vector2 near_bv= {0,0};
        bool found_bullet = false;
        int nearby_bullets = 0;
        Vector2 escape = {0,0};

        for (auto* b : e_bullets) {
            if (!is_visible(b->get_pos())) continue;
            float d = p_pos.distance_to(b->get_pos());
            
            if (d < 200.f) {
                nearby_bullets++;
                Vector2 away = (p_pos - b->get_pos());
                escape += away.normalize() / (d + 1.f);
            }

            if (d < min_bd) { min_bd = d; near_b = b->get_pos(); near_bv = b->get_vel(); found_bullet = true; }
        }

        if (found_bullet) {
            Vector2 b_dir = (near_b - p_pos).normalize();
            s.push_back(b_dir.x);
            s.push_back(b_dir.y);
            s.push_back(std::min(1.f, min_bd / 500.f));
            // Incoming bullet velocity direction (is it heading toward player?)
            Vector2 b_vel_n = near_bv.normalize();
            s.push_back(b_vel_n.x);
            s.push_back(b_vel_n.y);
        } else {
            s.push_back(0.f); s.push_back(0.f); // dir
            s.push_back(1.f);                   // max dist
            s.push_back(0.f); s.push_back(0.f); // vel dir
        }

        // Shoot cooldown readiness
        s.push_back(1.f - player->get_cd() / 10.f);

        // Player health
        s.push_back(player->get_health() / 100.f);

        // --- ENHANCED VISION (v2) ---
        // 17. Bullet density
        s.push_back(std::min(1.f, nearby_bullets / 10.f));

        // 18, 19. Escape direction hint
        escape = escape.normalize();
        s.push_back(escape.x);
        s.push_back(escape.y);

        // 20. Wall distance
        float wall_dist = std::min({
            p_pos.x,
            WIDTH - p_pos.x,
            p_pos.y,
            HEIGHT - p_pos.y
        }) / 300.f;
        s.push_back(std::min(1.f, wall_dist));

        assert((int)s.size() == STATE_DIM);
        return s;
    }

    // ── Predictive aim target for the nearest enemy ──
    Vector2 predict_aim(const Vector2& p_pos, const Vector2& e_pos, const Vector2& e_vel, float b_speed) {
        float dist = p_pos.distance_to(e_pos);
        // Iterative refinement (3 iterations)
        Vector2 pred = e_pos;
        for (int i = 0; i < 3; i++) {
            float t   = p_pos.distance_to(pred) / b_speed;
            pred      = e_pos + e_vel * t;
        }
        // Clamp: don't predict too far
        if (p_pos.distance_to(pred) > dist + 250.f) pred = e_pos;
        return pred;
    }

    void handle_events() {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) running = false;
            if (e.type == SDL_KEYDOWN) {
                keys[e.key.keysym.scancode] = 1;
                if (e.key.keysym.scancode == SDL_SCANCODE_ESCAPE) running = false;
            }
            if (e.type == SDL_KEYUP)   keys[e.key.keysym.scancode] = 0;
            if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_LEFT)  keys[SDL_SCANCODE_KP_0] = 1;
            if (e.type == SDL_MOUSEBUTTONUP   && e.button.button == SDL_BUTTON_LEFT)  keys[SDL_SCANCODE_KP_0] = 0;
        }
    }

    void reset() {
        for (auto* e : enemies)   delete e;
        for (auto* b : p_bullets) delete b;
        for (auto* b : e_bullets) delete b;
        for (auto* p : particles) delete p;
        enemies.clear(); p_bullets.clear(); e_bullets.clear(); particles.clear();

        delete player; player = new Player();
        if (player_tex) player->set_texture(player_tex);

        // Print accuracy
        float acc = shots_fired > 0 ? 100.f * shots_hit / shots_fired : 0.f;
        std::cout << "[Stats] Score=" << score
                  << " Accuracy=" << acc << "% (" << shots_hit << "/" << shots_fired << ")\n";

        score = 0; total_reward = 0; spawn_timer = 0; shake = 0;
        prev_min_b_dist = 1e9f;
        tracked_enemy_vel = {0,0};
        tracked_enemy_pos = {0,0};
        shots_fired = shots_hit = 0;

        if (ai_mode) net->save("model.weights");
        std::cout << "[Game] Reset\n";
    }

    void update() {
        player->update();
        bg->update();

        // Update enemy velocity tracking
        {
            float min_d = 1e9f;
            Enemy* near = nullptr;
            for (auto* e : enemies) {
                float d = player->get_pos().distance_to(e->get_pos());
                if (d < min_d) { min_d = d; near = e; }
            }
            if (near) {
                tracked_enemy_vel = near->get_vel_smooth();
                tracked_enemy_pos = near->get_pos();
            }
        }

        Vec state = get_state();
        Vec act(NUM_ACTIONS, 0.f);

        if (ai_mode) {
            act = net->forward(state, nullptr, false);

            // Gaussian exploration noise (decaying)
            for (int i = 0; i < 4; i++) {
                act[i] += RNG::normal(0, epsilon);
                act[i]  = std::clamp(act[i], -1.f, 1.f);
            }
            act[SHOOT_PROB] += RNG::normal(0, epsilon * 0.4f);
            act[SHOOT_PROB]  = std::clamp(act[SHOOT_PROB], 0.f, 1.f);

            epsilon = std::max(EPS_MIN, epsilon * EPS_DECAY);
        } else {
            // ── Human input ──
            if (keys[SDL_SCANCODE_W]) act[MOVE_Y] = -1.f;
            if (keys[SDL_SCANCODE_S]) act[MOVE_Y] =  1.f;
            if (keys[SDL_SCANCODE_A]) act[MOVE_X] = -1.f;
            if (keys[SDL_SCANCODE_D]) act[MOVE_X] =  1.f;

            int mx, my;
            SDL_GetMouseState(&mx, &my);
            Vector2 aim = (Vector2{(float)mx, (float)my} - player->get_pos()).normalize();
            act[AIM_X] = aim.x; act[AIM_Y] = aim.y;
            if (keys[SDL_SCANCODE_SPACE] || keys[SDL_SCANCODE_KP_0]) act[SHOOT_PROB] = 1.f;
        }

        // ── Move ──
        Vector2 move_dir = {act[MOVE_X], act[MOVE_Y]};
        player->move(move_dir, ai_mode);

        // ── Reward Calculation ──
        float reward = 0.03f + (step_count % 10000) * 0.00001f; // scaling survival bonus
        if (step_count > 2000) reward += 0.05f; // Late survival bonus
        Vector2 p_pos = player->get_pos();

        // Soft center gravity (quadratic)
        float center_d = p_pos.distance_to({WIDTH/2.f, HEIGHT/2.f});
        float norm_cd  = center_d / 450.f;
        reward -= norm_cd * norm_cd * 0.08f;

        // Boundary penalty (hard)
        if (ai_mode) {
            int cx = WIDTH/2, cy = HEIGHT/2;
            if (p_pos.x < cx-290 || p_pos.x > cx+290 || p_pos.y < cy-290 || p_pos.y > cy+290)
                reward -= 2.f;
        }

        // Edge stick penalty (time-based)
        bool near_edge = (p_pos.x < 100 || p_pos.x > WIDTH-100 || p_pos.y < 100 || p_pos.y > HEIGHT-100);
        if (near_edge) edge_timer++;
        else           edge_timer = 0;
        if (edge_timer > 60) reward -= 0.2f;

        // Re-detect nearest entities for rewards (now using vision)
        float  min_ed = 1e9f, min_bd = 1e9f;
        Vector2 near_ep = {0,0}, near_ev = {0,0}, near_bp = {0,0}, near_bv = {0,0};
        bool found_enemy = false, found_bullet = false;
        int nearby_bullets = 0;

        for (auto* e : enemies) {
            if (!is_visible(e->get_pos())) continue;
            float d = p_pos.distance_to(e->get_pos());
            if (d < min_ed) { min_ed = d; near_ep = e->get_pos(); near_ev = e->get_vel_smooth(); found_enemy = true; }
        }
        for (auto* b : e_bullets) {
            if (!is_visible(b->get_pos())) continue;
            float d = p_pos.distance_to(b->get_pos());
            if (d < 200.f) nearby_bullets++;
            if (d < min_bd) { min_bd = d; near_bp = b->get_pos(); near_bv = b->get_vel(); found_bullet = true; }
        }

        // Multi-bullet awareness
        reward -= nearby_bullets * 0.02f; // Softened bullet density penalty

        // Smart movement reward
        float move_mag = move_dir.length();
        if (move_mag < 0.05f) {
            reward -= 0.15f; // standing still = illegal
        } else {
            // Reward dodging if bullet is near
            if (found_bullet && min_bd < 250.f) {
                Vector2 away = (p_pos - near_bp).normalize();
                float smart_move = move_dir.normalize().dot(away);
                reward += smart_move * 0.4f; // Keep strong dodge signal
            } else {
                reward += 0.05f; // moving = good
            }
        }

        // Dodge trajectory reward & Panic Zone
        if (found_bullet && min_bd < 250.f) {
            if (min_bd < 150.f) reward -= 0.25f; // Softened Panic Zone
            if (min_bd > prev_min_b_dist) reward += 0.25f;
            Vector2 bdir  = near_bv.normalize();
            float   oncoming = -move_dir.normalize().dot(bdir);
            reward += oncoming * 0.3f; // buffed oncoming dodging
        }
        prev_min_b_dist = min_bd;

        // Enemy proximity penalty & engagement reward
        if (found_enemy) {
            if (min_ed < 110.f) reward -= 0.12f;
            if (min_ed < 350.f) reward += 0.05f; // Engagement reward
            reward += 0.5f * (1.0f / (1.0f + min_ed)); // Reward killing faster/proximity
        }

        // Accuracy reward
        float acc = shots_fired > 0 ? (float)shots_hit / shots_fired : 0.f;
        reward += acc * 0.5f;
        if (shots_fired > 50) reward += acc * 0.3f; // Protect accuracy

        // ── Shoot ──
        if (act[SHOOT_PROB] > 0.5f && player->can_shoot()) {
            if (found_enemy) {
                // Softened encouragement
                if (step_count < 20000) reward += 0.08f;
                reward -= 0.1f; // Ammo cost

                float b_speed = 10.f;
                Vector2 aim_dir = {act[AIM_X], act[AIM_Y]};

                if (ai_mode) {
                    // Compute predictive aim
                    Vector2 pred_target = predict_aim(p_pos, near_ep, near_ev, b_speed);
                    Vector2 pred_dir    = (pred_target - p_pos).normalize();

                    // Blend AI aim with prediction (60/40) for imperfection later 
                    // if (aim_dir.length() > 0.1f) {
                    //     aim_dir = (aim_dir.normalize() * 0.6f + pred_dir * 0.4f).normalize();
                    // } else {
                        aim_dir = {act[AIM_X], act[AIM_Y]};
                    // }

                    // Aim quality reward
                    Vector2 to_e  = (near_ep - p_pos).normalize();
                    float   aq    = aim_dir.dot(to_e);

                    // Update player angle to face aim direction
                    player->set_angle(std::atan2(aim_dir.y, aim_dir.x) * 180.0f / 3.14159265f);

if (aq > 0.95f)      reward += 2.5f;
else if (aq > 0.85f) reward += 1.2f;
else if (aq > 0.7f)  reward += 0.3f;
else                 reward -= 0.8f;

                    if (min_ed > 500.f) reward -= 0.3f;

                    Vector2 fire_target = p_pos + aim_dir * 1200.f;
                    p_bullets.push_back(new Bullet(p_pos, fire_target, {120, 255, 120, 255}, 10, b_speed));
                    player->fire();
                    shots_fired++;
                } else {
                    // Human: use mouse aim
                    if (aim_dir.length() < 0.1f)
                        aim_dir = (near_ep - p_pos).normalize();
                    else
                        aim_dir = aim_dir.normalize();

                    // Update player angle to face aim direction
                    player->set_angle(std::atan2(aim_dir.y, aim_dir.x) * 180.0f / 3.14159265f);

                    Vector2 fire_target = p_pos + aim_dir * 1200.f;
                    p_bullets.push_back(new Bullet(p_pos, fire_target, {120, 255, 120, 255}, 10, b_speed));
                    player->fire();
                    shots_fired++;
                }
            } else {
                reward -= 0.35f; // shooting with no target
            }
        } else {
            // Even if not shooting, update player rotation to face aim or nearest enemy
            Vector2 aim_dir = {act[AIM_X], act[AIM_Y]};
            if (ai_mode) {
                if (found_enemy) {
                    Vector2 pred_target = predict_aim(p_pos, near_ep, near_ev, 10.f);
                    Vector2 pred_dir    = (pred_target - p_pos).normalize();
                    player->set_angle(std::atan2(pred_dir.y, pred_dir.x) * 180.0f / 3.14159265f);
                }
            } else {
                if (aim_dir.length() > 0.1f)
                    player->set_angle(std::atan2(aim_dir.y, aim_dir.x) * 180.0f / 3.14159265f);
            }
        }

        reward = std::clamp(reward, -10.f, 10.f);

        // Update all entities; reward may be modified
        update_entities(reward);

        total_reward += reward;
        step_count++;

        if (ai_mode) {
            replay.push(state, act, reward);
            if (step_count % train_every == 0 && replay.size() >= 256) {
                std::vector<Vec> s_batch, a_batch;
                std::vector<float> r_batch;
                if (replay.sample(256, s_batch, a_batch, r_batch)) {
                    net->train_batch(s_batch, a_batch, r_batch);
                    float avg_r = 0; for (float r : r_batch) avg_r += r;
                    avg_r /= r_batch.size();
                    std::cout << "[Train] step=" << step_count
                              << " eps=" << epsilon
                              << " avg_r=" << avg_r
                              << " buf=" << replay.size() << "\n";
                }
            }
            if (step_count % 2000 == 0) net->save("model.weights");
        }

        // Spawn enemies
        int max_enemies = std::min(8, 2 + score/500);
        int spawn_interval = std::max(20, 80 - score/300);
        if (++spawn_timer >= spawn_interval && (int)enemies.size() < max_enemies) {
            Enemy* e = new Enemy(player, e_bullets);
            if (enemy_tex) e->set_texture(enemy_tex);
            enemies.push_back(e);
            spawn_timer = 0;
        }
    }

    void update_entities(float& reward) {
        // Enemies
        for (auto eit = enemies.begin(); eit != enemies.end();) {
            (*eit)->update();
            bool killed = false;
            SDL_Rect er = (*eit)->get_rect();

            for (auto bit = p_bullets.begin(); bit != p_bullets.end();) {
                SDL_Rect br = (*bit)->get_rect();
                if (SDL_HasIntersection(&er, &br)) {
                    (*eit)->take_damage(50);
                    reward     += 8.f;
                    shots_hit  += 1;
                    delete *bit; bit = p_bullets.erase(bit);
                    if (!(*eit)->is_alive() && !killed) {
                        spawn_particles((*eit)->get_pos());
                        score   += 100;
                        reward  += 100.f;
                        killed   = true;
                    }
                } else ++bit;
            }

            if (killed) { delete *eit; eit = enemies.erase(eit); continue; }

            // Player collision with enemy body
            SDL_Rect pr = player->get_rect();
            if (SDL_HasIntersection(&er, &pr)) {
                reward -= 5.f;
                std::cout << "[Game] Over — collided with enemy\n";
                if (ai_mode) {
                    std::vector<Vec> sb, ab; std::vector<float> rb;
                    if (replay.sample(128, sb, ab, rb)) net->train_batch(sb, ab, rb);
                }
                reset(); return;
            }
            ++eit;
        }

        // Player bullets (near-miss shaping)
        for (auto bit = p_bullets.begin(); bit != p_bullets.end();) {
            (*bit)->update();
            if (!(*bit)->is_alive()) { 
                delete *bit; 
                bit = p_bullets.erase(bit); 
            }
            else ++bit;
        }

        // Enemy bullets
        for (auto bit = e_bullets.begin(); bit != e_bullets.end();) {
            (*bit)->update();
            SDL_Rect br = (*bit)->get_rect(), pr = player->get_rect();
            if (SDL_HasIntersection(&br, &pr)) {
                player->take_damage(40);
                reward   -= 2.5f;
                shake     = 16; // Stronger shake
                delete *bit; bit = e_bullets.erase(bit);
                if (player->get_health() <= 0) {
                    reward -= 5.f;
                    std::cout << "[Game] Over — HP zero\n";
                    if (ai_mode) {
                        std::vector<Vec> sb, ab; std::vector<float> rb;
                        if (replay.sample(128, sb, ab, rb)) net->train_batch(sb, ab, rb);
                    }
                    reset(); return;
                }
            } else if (!(*bit)->is_alive()) { delete *bit; bit = e_bullets.erase(bit); }
            else ++bit;
        }

        // Particles
        for (auto pit = particles.begin(); pit != particles.end();) {
            (*pit)->update();
            if (!(*pit)->is_alive()) { delete *pit; pit = particles.erase(pit); }
            else ++pit;
        }
    }

    void spawn_particles(Vector2 p) {
        SDL_Color colors[] = {{255,80,80,255},{255,160,60,255},{255,255,100,255}};
        for (int i = 0; i < 14; i++)
            particles.push_back(new Particle(p, colors[RNG::rng_int(0,2)]));
    }

    void render() {
        int ox = 0, oy = 0;
        if (shake > 0) {
            ox = RNG::rng_int(-shake, shake);
            oy = RNG::rng_int(-shake, shake);
            shake--;
        }
        SDL_SetRenderDrawColor(renderer, 20, 20, 28, 255);
        SDL_RenderClear(renderer);
        bg->draw(renderer, ox, oy);

        // AI play-zone indicator
        if (ai_mode) {
            SDL_SetRenderDrawColor(renderer, 80, 80, 160, 60);
            int cx = WIDTH/2, cy = HEIGHT/2;
            SDL_Rect zone = {cx-300+ox, cy-300+oy, 600, 600};
            SDL_RenderDrawRect(renderer, &zone);
        }

        for (auto* p : particles)  p->draw(renderer, ox, oy);
        for (auto* b : p_bullets)  b->draw(renderer, ox, oy);
        for (auto* b : e_bullets)  b->draw(renderer, ox, oy);
        for (auto* e : enemies) {
            e->draw(renderer, ox, oy);
            char buf[8]; sprintf(buf, "%d", e->get_health());
            render_text(buf, (int)e->get_pos().x+ox, (int)e->get_pos().y-32+oy, {255,100,100,255}, true);
        }
        player->draw(renderer, ox, oy);
        draw_hud();
        SDL_RenderPresent(renderer);
    }

    void render_text(const char* txt, int x, int y, SDL_Color c, bool center = false) {
        if (!font) return;
        SDL_Surface* surf = TTF_RenderText_Solid(font, txt, c);
        if (!surf) return;
        SDL_Texture* tex = SDL_CreateTextureFromSurface(renderer, surf);
        SDL_Rect dst = {x - (center ? surf->w/2 : 0), y, surf->w, surf->h};
        SDL_RenderCopy(renderer, tex, NULL, &dst);
        SDL_FreeSurface(surf);
        SDL_DestroyTexture(tex);
    }

    void draw_hud() {
        if (!font) return;
        char buf[128];
        float acc = shots_fired > 0 ? 100.f * shots_hit / shots_fired : 0.f;
        
        SDL_Color hp_color = {200, 255, 200, 255};
        if (player->get_health() < 30) hp_color = {255, 80, 80, 255};
        else if (player->get_health() < 60) hp_color = {255, 255, 100, 255};

        sprintf(buf, "HP: %d",        player->get_health()); render_text(buf, 12, 12,  hp_color);
        sprintf(buf, "Score: %d",     score);                render_text(buf, 12, 40,  {120, 200, 255, 255});
        sprintf(buf, "Reward: %.1f",  total_reward);         render_text(buf, 12, 68,  {180, 240, 180, 255});
        sprintf(buf, "Acc: %.1f%%",   acc);                  render_text(buf, 12, 96,  {255, 220, 80, 255});
        sprintf(buf, "Eps: %.3f",     epsilon);              render_text(buf, 12, 124, {160, 200, 255, 255});
        sprintf(buf, "Buf: %d",       replay.size());        render_text(buf, 12, 152, {160, 200, 255, 255});
        if (!ai_mode) render_text("HUMAN MODE", WIDTH-160, 12, {255,200,80,255});
        else          render_text("AI MODE",    WIDTH-130, 12, {80,220,255,255});
    }

    void cleanup() {
        if (net) net->save("model.weights");
        delete player; delete net; delete bg;
        if (player_tex) SDL_DestroyTexture(player_tex);
        if (enemy_tex)  SDL_DestroyTexture(enemy_tex);
        for (auto* e : enemies)   delete e;
        for (auto* b : p_bullets) delete b;
        for (auto* b : e_bullets) delete b;
        for (auto* p : particles) delete p;
        if (font)     TTF_CloseFont(font);
        if (renderer) SDL_DestroyRenderer(renderer);
        if (window)   SDL_DestroyWindow(window);
        TTF_Quit(); IMG_Quit(); SDL_Quit();
    }
};

// ─────────────────────────────────────────────
int main(int argc, char* argv[]) {
    bool ai = true;
    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "h" || arg == "human") ai = false;
    }
    Game game;
    if (game.init(ai)) game.run();
    return 0;
}
