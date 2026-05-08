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
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <memory>
#include <iomanip>

// ─────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────
const int WIDTH  = 1600;
const int HEIGHT = 900;
static constexpr int   N_ENVS       = 32;
static constexpr int   PPO_EPOCHS   = 4;
static constexpr int   BATCH_SIZE   = 256;
static constexpr float GAMMA        = 0.99f;
static constexpr float LAMBDA_GAE   = 0.95f;
static constexpr float PPO_CLIP     = 0.2f;
static constexpr float ENTROPY_BETA = 0.01f;
static constexpr float VALUE_LOSS_COEFF = 0.5f;

// Action indices
enum Action { MOVE_X = 0, MOVE_Y, AIM_X, AIM_Y, SHOOT_PROB, NUM_ACTIONS };

// ─────────────────────────────────────────────────────────────
// Math Utilities
// ─────────────────────────────────────────────────────────────
template<typename T>
void clear_container(std::vector<T*>& vec) {
    for (auto* p : vec) delete p;
    vec.clear();
}

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

// Thread-local RNG for parallel environments
static thread_local std::mt19937 g_rng{std::random_device{}()};
class RNG {
public:
    static float flt(float lo, float hi) {
        return std::uniform_real_distribution<float>(lo, hi)(g_rng);
    }
    static int rng_int(int lo, int hi) {
        return std::uniform_int_distribution<int>(lo, hi)(g_rng);
    }
    static float normal(float mean = 0.f, float std = 1.f) {
        return std::normal_distribution<float>(mean, std)(g_rng);
    }
};

// ─────────────────────────────────────────────────────────────
// Linear Algebra Helpers
// ─────────────────────────────────────────────────────────────
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

// FIX: d_leaky_relu operates on pre-layer-norm activations.
// Previously the cache stored post-layer-norm values but they were
// named "p1/p2/p3" — now we explicitly store the pre-norm values
// (before layer_norm is applied) so the gradient is correct.
static Vec d_leaky_relu(const Vec& pre_norm, float alpha = 0.01f) {
    Vec out(pre_norm.size());
    for (int i = 0; i < (int)pre_norm.size(); i++)
        out[i] = pre_norm[i] >= 0.f ? 1.f : alpha;
    return out;
}

// FIX: layer_norm now returns the pre-norm vector so callers can
// cache it for correct backward-pass gradient computation.
static Vec layer_norm_inplace(Vec& v, float eps = 1e-5f) {
    Vec pre = v; // capture before normalising
    float mean = 0, var = 0;
    for (float x : v) mean += x;
    mean /= static_cast<float>(v.size());
    for (float x : v) var += (x - mean) * (x - mean);
    var /= static_cast<float>(v.size());
    float inv = 1.f / std::sqrt(var + eps);
    for (float& x : v) x = (x - mean) * inv;
    return pre; // caller stores this for d_leaky_relu
}

struct AdamState {
    Mat m, v;
    Vec mb, vb;
    int t = 0;
    AdamState() = default;
    AdamState(int rows, int cols)
        : m(make_mat(rows, cols)), v(make_mat(rows, cols)),
          mb(rows, 0.f), vb(rows, 0.f) {}
};

static void adam_update(Mat& W, Vec& b, const Mat& dW, const Vec& db,
                        AdamState& st, float lr, float beta1 = 0.9f,
                        float beta2 = 0.999f, float eps = 1e-8f)
{
    st.t++;
    float bc1 = 1.f - std::pow(beta1, st.t);
    float bc2 = 1.f - std::pow(beta2, st.t);
    for (int i = 0; i < (int)W.size(); i++) {
        for (int j = 0; j < (int)W[0].size(); j++) {
            st.m[i][j] = beta1 * st.m[i][j] + (1.f-beta1) * dW[i][j];
            st.v[i][j] = beta2 * st.v[i][j] + (1.f-beta2) * dW[i][j] * dW[i][j];
            W[i][j] += lr * (st.m[i][j] / bc1) / (std::sqrt(st.v[i][j] / bc2) + eps);
        }
        st.mb[i] = beta1 * st.mb[i] + (1.f-beta1) * db[i];
        st.vb[i] = beta2 * st.vb[i] + (1.f-beta2) * db[i] * db[i];
        b[i] += lr * (st.mb[i] / bc1) / (std::sqrt(st.vb[i] / bc2) + eps);
    }
}

// ─────────────────────────────────────────────────────────────
// Actor-Critic Network (PPO)
// ─────────────────────────────────────────────────────────────
static constexpr int STATE_DIM = 22;
static constexpr int H1 = 128, H2 = 128, H3 = 64;

struct ActorCriticNet {
    Mat w1, w2, w3, w4;
    Vec b1, b2, b3, b4;
    // Old weights for PPO ratio — updated only via snapshot()
    Mat ow1, ow2, ow3, ow4;
    Vec ob1, ob2, ob3, ob4;
    AdamState as1, as2, as3, as4;
    float lr = 2e-4f;

    ActorCriticNet() {
        auto he = [](int fi) { return RNG::normal(0, std::sqrt(2.f / static_cast<float>(fi))); };
        auto init = [&](Mat& W, Vec& b, int r, int c) {
            W = make_mat(r, c); b.assign(r, 0.f);
            for (auto& row : W) for (auto& w : row) w = he(c);
        };
        init(w1, b1, H1, STATE_DIM);
        init(w2, b2, H2, H1);
        init(w3, b3, H3, H2);
        init(w4, b4, NUM_ACTIONS + 1, H3); // +1 for Value head
        b4[SHOOT_PROB] = -1.f;
        as1 = AdamState(H1, STATE_DIM);
        as2 = AdamState(H2, H1);
        as3 = AdamState(H3, H2);
        as4 = AdamState(NUM_ACTIONS + 1, H3);
        snapshot();
    }

    // FIX: Cache now stores true pre-normalisation values so that
    // d_leaky_relu receives the correct inputs during backprop.
    struct Cache {
        Vec h1, h2, h3;       // post-activation
        Vec pre1, pre2, pre3; // pre-layer-norm (for gradient)
    };

    std::pair<Vec, float> forward(const Vec& s, Cache* c = nullptr) const {
        Vec z1 = mat_vec_mul(w1, s); add_bias(z1, b1);
        Vec pre1 = z1;               // save before layer_norm
        layer_norm_inplace(z1);
        Vec h1 = leaky_relu(z1);

        Vec z2 = mat_vec_mul(w2, h1); add_bias(z2, b2);
        Vec pre2 = z2;
        layer_norm_inplace(z2);
        Vec h2 = leaky_relu(z2);

        Vec z3 = mat_vec_mul(w3, h2); add_bias(z3, b3);
        Vec pre3 = z3;
        layer_norm_inplace(z3);
        Vec h3 = leaky_relu(z3);

        Vec out = mat_vec_mul(w4, h3); add_bias(out, b4);

        if (c) *c = {h1, h2, h3, pre1, pre2, pre3};

        Vec act(NUM_ACTIONS);
        for (int i = 0; i < 4; i++) act[i] = std::tanh(out[i]);
        act[SHOOT_PROB] = 1.f / (1.f + std::exp(-out[SHOOT_PROB]));
        return {act, out[NUM_ACTIONS]};
    }

    Vec forward_old(const Vec& s) const {
        Vec z1 = mat_vec_mul(ow1, s); add_bias(z1, ob1); layer_norm_inplace(z1); Vec h1 = leaky_relu(z1);
        Vec z2 = mat_vec_mul(ow2, h1);add_bias(z2, ob2); layer_norm_inplace(z2); Vec h2 = leaky_relu(z2);
        Vec z3 = mat_vec_mul(ow3, h2);add_bias(z3, ob3); layer_norm_inplace(z3); Vec h3 = leaky_relu(z3);
        Vec out = mat_vec_mul(ow4, h3); add_bias(out, ob4);
        Vec act(NUM_ACTIONS);
        for (int i = 0; i < 4; i++) act[i] = std::tanh(out[i]);
        act[SHOOT_PROB] = 1.f / (1.f + std::exp(-out[SHOOT_PROB]));
        return act;
    }

    void snapshot() {
        ow1=w1; ow2=w2; ow3=w3; ow4=w4;
        ob1=b1; ob2=b2; ob3=b3; ob4=b4;
    }

    void copy_from(const ActorCriticNet& o) {
        w1=o.w1; w2=o.w2; w3=o.w3; w4=o.w4;
        b1=o.b1; b2=o.b2; b3=o.b3; b4=o.b4;
    }

    // FIX: ppo_update now accepts pre-computed old log-probs collected at
    // rollout time, avoiding re-running forward_old on every training sample
    // and preventing stale ratio computation if snapshot() were ever called
    // mid-update.
    void ppo_update(const std::vector<Vec>& states,
                    const std::vector<Vec>& actions,
                    const std::vector<float>& advantages,
                    const std::vector<float>& returns,
                    const std::vector<float>& old_log_probs,
                    const std::vector<float>& epsilons,
                    std::mutex& mtx,
                    const std::atomic<bool>& running)
    {
        int N = (int)states.size();
        std::vector<int> idx(N); std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), g_rng);

        for (int bs = 0; bs < N; bs += BATCH_SIZE) {
            if (!running.load()) break;
            int be = std::min(bs + BATCH_SIZE, N);
            Mat dw1 = make_mat(H1, STATE_DIM), dw2 = make_mat(H2, H1),
                dw3 = make_mat(H3, H2),        dw4 = make_mat(NUM_ACTIONS + 1, H3);
            Vec db1(H1, 0), db2(H2, 0), db3(H3, 0), db4(NUM_ACTIONS + 1, 0);

            for (int bi = bs; bi < be; bi++) {
                int i = idx[bi];
                Cache c; auto [cur_act, cur_val] = forward(states[i], &c);

                // Compute log-prob of current policy
                float log_new = 0.f;
                for (int k = 0; k < NUM_ACTIONS; k++) {
                    float pn  = cur_act[k];
                    float act = actions[i][k];
                    if (k == SHOOT_PROB) {
                        log_new += act * std::log(pn + 1e-8f) + (1.f - act) * std::log(1.f - pn + 1e-8f);
                    } else {
                        float sig = std::max(0.05f, epsilons[i]);
                        log_new += -0.5f * std::pow((act - pn) / sig, 2.f);
                    }
                }

                // FIX: use pre-computed old log-prob from rollout
                float ratio = std::exp(std::clamp(log_new - old_log_probs[i], -5.f, 5.f));
                float A = advantages[i];
                float clip_r = std::clamp(ratio, 1.f - PPO_CLIP, 1.f + PPO_CLIP);
                bool clipped = (A > 0.f && ratio > 1.f + PPO_CLIP) ||
                               (A < 0.f && ratio < 1.f - PPO_CLIP);
                float eff = clipped ? clip_r : ratio;

                Vec out_err(NUM_ACTIONS + 1, 0.f);
                for (int k = 0; k < NUM_ACTIONS; k++) {
                    float pn  = cur_act[k];
                    float act = actions[i][k];
                    if (k == SHOOT_PROB) {
                        out_err[k] = eff * A * (act - pn);
                        float logit = std::log(pn / (1.f - pn + 1e-8f) + 1e-8f);
                        out_err[k] -= ENTROPY_BETA * logit * pn * (1.f - pn);
                    } else {
                        float sig = std::max(0.05f, epsilons[i]);
                        out_err[k] = eff * A * (act - pn) / (sig * sig) * (1.f - pn * pn);
                    }
                }
                out_err[NUM_ACTIONS] = (returns[i] - cur_val) * VALUE_LOSS_COEFF;

                // Backprop through layer 4 → 3
                Vec h3_err(H3, 0.f);
                for (int j = 0; j < H3; j++) {
                    for (int k = 0; k < NUM_ACTIONS + 1; k++) {
                        dw4[k][j] += out_err[k] * c.h3[j];
                        h3_err[j] += w4[k][j] * out_err[k];
                    }
                }
                for (int k = 0; k < NUM_ACTIONS + 1; k++) db4[k] += out_err[k];

                // FIX: use pre-norm cache (pre3/pre2/pre1) for d_leaky_relu
                Vec d3 = d_leaky_relu(c.pre3); Vec h2_err(H2, 0.f);
                for (int j = 0; j < H2; j++) {
                    for (int k = 0; k < H3; k++) {
                        float g = h3_err[k] * d3[k];
                        dw3[k][j] += g * c.h2[j];
                        h2_err[j] += w3[k][j] * g;
                    }
                }
                for (int k = 0; k < H3; k++) db3[k] += h3_err[k] * d3[k];

                Vec d2 = d_leaky_relu(c.pre2); Vec h1_err(H1, 0.f);
                for (int j = 0; j < H1; j++) {
                    for (int k = 0; k < H2; k++) {
                        float g = h2_err[k] * d2[k];
                        dw2[k][j] += g * c.h1[j];
                        h1_err[j] += w2[k][j] * g;
                    }
                }
                for (int k = 0; k < H2; k++) db2[k] += h2_err[k] * d2[k];

                Vec d1 = d_leaky_relu(c.pre1);
                for (int j = 0; j < STATE_DIM; j++) {
                    for (int k = 0; k < H1; k++) {
                        dw1[k][j] += h1_err[k] * d1[k] * states[i][j];
                    }
                }
                for (int k = 0; k < H1; k++) db1[k] += h1_err[k] * d1[k];
            }

            float sc = 1.f / static_cast<float>(be - bs);
            auto nrm = [&](Mat& M, Vec& v) {
                for (auto& r : M) for (auto& x : r) x *= sc;
                for (auto& x : v) x *= sc;
            };
            nrm(dw1, db1); nrm(dw2, db2); nrm(dw3, db3); nrm(dw4, db4);

            std::lock_guard<std::mutex> lk(mtx);
            adam_update(w1, b1, dw1, db1, as1, lr);
            adam_update(w2, b2, dw2, db2, as2, lr);
            adam_update(w3, b3, dw3, db3, as3, lr);
            adam_update(w4, b4, dw4, db4, as4, lr);
        }
    }

    void save(const std::string& p) const {
        std::ofstream f(p, std::ios::binary); if (!f) return;
        auto wm = [&](const Mat& M) { for (const auto& r : M) f.write(reinterpret_cast<const char*>(r.data()), static_cast<std::streamsize>(r.size() * 4)); };
        auto wv = [&](const Vec& v) { f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * 4)); };
        wm(w1); wm(w2); wm(w3); wm(w4); wv(b1); wv(b2); wv(b3); wv(b4);
    }

    void load(const std::string& p) {
        std::ifstream f(p, std::ios::binary); if (!f) return;
        auto rm = [&](Mat& M) { for (auto& r : M) f.read(reinterpret_cast<char*>(r.data()), static_cast<std::streamsize>(r.size() * 4)); };
        auto rv = [&](Vec& v) { f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * 4)); };
        rm(w1); rm(w2); rm(w3); rm(w4); rv(b1); rv(b2); rv(b3); rv(b4); snapshot();
    }
};

// ─────────────────────────────────────────────────────────────
// Experience Buffer (PPO)
// FIX: old_log_prob is now stored at collection time so train loop
// doesn't re-run forward_old across episode boundaries.
// ─────────────────────────────────────────────────────────────
struct Exp {
    Vec   s, a;
    float r, v, old_log_prob, eps;
    bool  done;
    Vec   ns;
};
using Episode = std::vector<Exp>;

class ReplayBuffer {
    // FIX: use std::deque so pop-front is O(1)
    std::deque<Episode> episodes;
    std::mutex mtx;
public:
    void push(Episode ep) {
        std::lock_guard<std::mutex> lk(mtx);
        episodes.push_back(std::move(ep));
        if (episodes.size() > 500) episodes.pop_front();
    }
    std::deque<Episode> swap_out() {
        std::lock_guard<std::mutex> lk(mtx);
        std::deque<Episode> out; out.swap(episodes);
        return out;
    }
    int size() { std::lock_guard<std::mutex> lk(mtx); return (int)episodes.size(); }
};

// ─────────────────────────────────────────────────────────────
// Entities
// ─────────────────────────────────────────────────────────────
class Entity {
public:
    Vector2 pos;
    int w, h;
    SDL_Color color;
    bool alive = true;
    Entity(Vector2 p, int w, int h, SDL_Color c) : pos(p), w(w), h(h), color(c) {}
    virtual ~Entity() = default;
    virtual void update() = 0;
    virtual void draw(SDL_Renderer* r, int ox, int oy) {
        if (!r) return;
        SDL_SetRenderDrawBlendMode(r, SDL_BLENDMODE_BLEND);
        SDL_SetRenderDrawColor(r, color.r, color.g, color.b, 60);
        SDL_Rect glow = { (int)std::round(pos.x - w/2) - 2 + ox, (int)std::round(pos.y - h/2) - 2 + oy, w + 4, h + 4 };
        SDL_RenderFillRect(r, &glow);
        SDL_SetRenderDrawColor(r, color.r, color.g, color.b, color.a);
        SDL_Rect rc = { (int)std::round(pos.x - w/2) + ox, (int)std::round(pos.y - h/2) + oy, w, h };
        SDL_RenderFillRect(r, &rc);
    }
    SDL_Rect get_rect() const { return { (int)(pos.x-w/2), (int)(pos.y-h/2), w, h }; }
};

class Particle : public Entity {
    Vector2 vel; int life;
public:
    Particle(Vector2 p, SDL_Color c) : Entity(p, 4, 4, c), vel(RNG::flt(-3,3), RNG::flt(-3,3)), life(30) {}
    void update() override { pos += vel; if (--life <= 0) alive = false; }
};

class Bullet : public Entity {
public:
    Vector2 vel;
    Bullet(Vector2 p, Vector2 target, SDL_Color c, int sz, float spd) : Entity(p, sz, sz, c) {
        vel = (target - p).normalize() * spd;
    }
    void update() override {
        pos += vel;
        if (pos.x < -50 || pos.x > WIDTH+50 || pos.y < -50 || pos.y > HEIGHT+50) alive = false;
    }
};

class Player : public Entity {
public:
    Vector2 velocity; int health = 100, shoot_cd = 0; float speed = 6.f;
    Player() : Entity({WIDTH/2.f, HEIGHT/2.f}, 28, 28, {200, 220, 255, 255}) {}
    void update() override { if (shoot_cd > 0) shoot_cd--; }
    void move(Vector2 dir, bool restricted = false) {
        if (dir.length() > 0) {
            if (dir.length() > 1.f) dir = dir.normalize();
            velocity = dir * speed; pos += velocity;
        } else velocity = {0,0};
        if (restricted) {
            int cx = WIDTH/2, cy = HEIGHT/2;
            pos.x = std::clamp(pos.x, static_cast<float>(cx-300+w/2), static_cast<float>(cx+300-w/2));
            pos.y = std::clamp(pos.y, static_cast<float>(cy-300+h/2), static_cast<float>(cy+300-h/2));
        } else {
            pos.x = std::clamp(pos.x, static_cast<float>(w/2), static_cast<float>(WIDTH-w/2));
            pos.y = std::clamp(pos.y, static_cast<float>(h/2), static_cast<float>(HEIGHT-h/2));
        }
    }
    bool can_shoot() const { return shoot_cd == 0; }
    void fire() { shoot_cd = 8; }
};

class Enemy : public Entity {
public:
    Player* player_ptr; std::vector<Bullet*>& bullets_ref;
    float spiral_speed, min_dist; int health = 150, shoot_cd;
    Vector2 prev_pos, vel_smooth;

    Enemy(Player* p, std::vector<Bullet*>& eb)
        : Entity({0,0}, 26, 26, {255, 80, 80, 255}), player_ptr(p), bullets_ref(eb), min_dist(160.f)
    {
        int side = RNG::rng_int(0,3);
        if      (side==0) pos = {RNG::flt(0,WIDTH), -35};
        else if (side==1) pos = {static_cast<float>(WIDTH)+35, RNG::flt(0,HEIGHT)};
        else if (side==2) pos = {RNG::flt(0,WIDTH), static_cast<float>(HEIGHT)+35};
        else              pos = {-35, RNG::flt(0,HEIGHT)};
        spiral_speed = RNG::flt(1.5f, 4.5f) * (RNG::rng_int(0,1) ? 1 : -1) / 100.f;
        shoot_cd = RNG::rng_int(120, 240); prev_pos = pos; vel_smooth = {0,0};
    }

    void update() override {
        Vector2 to_p = player_ptr->pos - pos; float dist = to_p.length(); Vector2 dir = to_p.normalize();
        if (dist > min_dist) pos += dir * 2.2f; else if (dist < min_dist-20.f) pos -= dir * 1.1f;
        Vector2 perp = {-dir.y, dir.x}; pos += perp * (spiral_speed * 140.f);
        
        // Clamp to screen
        pos.x = std::clamp(pos.x, (float)w/2, (float)WIDTH - w/2);
        pos.y = std::clamp(pos.y, (float)h/2, (float)HEIGHT - h/2);

        Vector2 raw = pos - prev_pos; vel_smooth = vel_smooth * 0.7f + raw * 0.3f; prev_pos = pos;
        if (--shoot_cd <= 0) {
            float bspd = 5.2f; float t = dist / bspd;
            Vector2 pred = player_ptr->pos + player_ptr->velocity * t;
            bullets_ref.push_back(new Bullet(pos, pred, {255, 140, 80, 255}, 8, bspd));
            shoot_cd = RNG::rng_int(80, 160);
        }
    }
};

// ─────────────────────────────────────────────────────────────
// Shared Simulation Helper
// ─────────────────────────────────────────────────────────────
static Vec build_state(Player* player, const std::vector<Enemy*>& enemies, const std::vector<Bullet*>& e_bullets) {
    Vec s;
    s.push_back(player->pos.x / WIDTH); s.push_back(player->pos.y / HEIGHT);
    s.push_back(player->velocity.x / 8.f); s.push_back(player->velocity.y / 8.f);

    std::vector<Enemy*> sorted = enemies;
    std::sort(sorted.begin(), sorted.end(), [&](Enemy* a, Enemy* b) {
        return player->pos.distance_to(a->pos) < player->pos.distance_to(b->pos);
    });

    for (int i = 0; i < 2; i++) {
        if (i < (int)sorted.size()) {
            Enemy* e = sorted[i];
            float d = player->pos.distance_to(e->pos);
            Vector2 e_rel = (e->pos - player->pos).normalize();
            s.push_back(e_rel.x); s.push_back(e_rel.y); s.push_back(std::min(1.f, d / 900.f));
            s.push_back(e->vel_smooth.x / 5.f); s.push_back(e->vel_smooth.y / 5.f);
        } else { for (int j = 0; j < 5; j++) s.push_back(j == 2 ? 1.f : 0.f); }
    }
    s.push_back(std::min(1.f, static_cast<float>(enemies.size()) / 10.f));

    float min_bd = 1e9f; Vector2 near_bp, near_bv; bool fb = false;
    for (auto* b : e_bullets) {
        float d = player->pos.distance_to(b->pos);
        if (d < min_bd) { min_bd = d; near_bp = b->pos; near_bv = b->vel; fb = true; }
    }
    if (fb) {
        Vector2 b_rel = (near_bp - player->pos).normalize();
        s.push_back(b_rel.x); s.push_back(b_rel.y); s.push_back(std::min(1.f, min_bd / 500.f));
        Vector2 bv = near_bv.normalize(); s.push_back(bv.x); s.push_back(bv.y);
    } else { for(int i=0; i<5; i++) s.push_back(i==2?1.f:0.f); }
    s.push_back(1.f - static_cast<float>(player->shoot_cd) / 10.f);
    s.push_back(static_cast<float>(player->health) / 100.f);
    return s;
}

static void perform_sim_step(Player* player, std::vector<Enemy*>& enemies,
                             std::vector<Bullet*>& p_bullets, std::vector<Bullet*>& e_bullets,
                             std::vector<Particle*>* particles,
                             int& score, float& reward, bool& done, std::atomic<long long>* hits = nullptr)
{
    // Enemies vs Player Bullets
    for (auto eit = enemies.begin(); eit != enemies.end();) {
        (*eit)->update(); bool killed = false; SDL_Rect er = (*eit)->get_rect();
        for (auto bit = p_bullets.begin(); bit != p_bullets.end();) {
            SDL_Rect br = (*bit)->get_rect();
            if (SDL_HasIntersection(&er, &br)) {
                (*eit)->health -= 50;
                // FIX: hit reward scaled to same order as death penalty
                reward += 5.f;
                if (hits) hits->fetch_add(1);
                delete *bit; bit = p_bullets.erase(bit);
                if ((*eit)->health <= 0) {
                    if (particles) {
                        for(int i=0; i<12; i++)
                            particles->push_back(new Particle((*eit)->pos, {255,100,100,255}));
                    }
                    score += 100;
                    reward += 20.f; // kill reward
                    killed = true;
                }
            } else {
                float d = (*eit)->pos.distance_to((*bit)->pos);
                if (d < 60.f) reward += 0.01f; // near-miss bonus, kept small
                ++bit;
            }
        }
        if (killed) { delete *eit; eit = enemies.erase(eit); continue; }
        SDL_Rect pr = player->get_rect();
        if (SDL_HasIntersection(&er, &pr)) {
            reward -= 20.f; done = true;
        }
        if (done) break;
        ++eit;
    }

    if (!done) {
        for (auto bit = p_bullets.begin(); bit != p_bullets.end();) {
            (*bit)->update(); if (!(*bit)->alive) { delete *bit; bit = p_bullets.erase(bit); } else ++bit;
        }
        for (auto bit = e_bullets.begin(); bit != e_bullets.end();) {
            (*bit)->update(); SDL_Rect br = (*bit)->get_rect(), pr = player->get_rect();
            if (SDL_HasIntersection(&br, &pr)) {
                player->health -= 40;
                // FIX: hit penalty scaled consistently
                reward -= 10.f;
                delete *bit; bit = e_bullets.erase(bit);
                if (player->health <= 0) { reward -= 20.f; done = true; }
            } else if (!(*bit)->alive) { delete *bit; bit = e_bullets.erase(bit); } else ++bit;
        }
    }

    if (particles) {
        for (auto pit = particles->begin(); pit != particles->end();) {
            (*pit)->update(); if (!(*pit)->alive) { delete *pit; pit = particles->erase(pit); } else ++pit;
        }
    }

    // FIX: reward shaping — removed aggression bonus that encouraged
    // suicidal rushing.  Survival + health bonus only.
    if (!done) {
        reward += 0.05f;                                          // survival
        reward += 0.005f * (static_cast<float>(player->health) / 100.f); // health
    }
}

// ─────────────────────────────────────────────────────────────
// Compute log-prob of an action under the current policy
// Used at rollout time to store old_log_prob in the experience.
// ─────────────────────────────────────────────────────────────
static float compute_log_prob(const Vec& act, const Vec& sampled_act, float eps_val) {
    float lp = 0.f;
    for (int k = 0; k < NUM_ACTIONS; k++) {
        float pn  = act[k];
        float a   = sampled_act[k];
        if (k == SHOOT_PROB) {
            lp += a * std::log(pn + 1e-8f) + (1.f - a) * std::log(1.f - pn + 1e-8f);
        } else {
            float sig = std::max(0.05f, eps_val);
            lp += -0.5f * std::pow((a - pn) / sig, 2.f);
        }
    }
    return lp;
}

// ─────────────────────────────────────────────────────────────
// Headless Environment struct
// ─────────────────────────────────────────────────────────────
struct Env {
    ActorCriticNet *shared_net;
    ActorCriticNet local_net;
    std::mutex *net_mtx;
    ReplayBuffer *buffer;
    std::atomic<float> *epsilon_ptr, *ai_control_ptr;
    std::atomic<long long> *total_shots_ptr, *total_hits_ptr;
    std::atomic<bool> *running_ptr;
    std::mutex *print_mtx;
    int id; bool is_headless;

    Player* player;
    std::vector<Enemy*> enemies;
    std::vector<Bullet*> p_bullets, e_bullets;
    int score = 0, spawn_timer = 0, step_count = 0;
    float total_reward = 0.f;

    Env(int i, ActorCriticNet* n, std::mutex* m, ReplayBuffer* b,
        std::atomic<float>* eps, std::atomic<float>* aic,
        std::atomic<long long>* ts, std::atomic<long long>* th,
        std::atomic<bool>* run, std::mutex* pmtx, bool hl)
        : shared_net(n), net_mtx(m), buffer(b), epsilon_ptr(eps),
          ai_control_ptr(aic), total_shots_ptr(ts), total_hits_ptr(th),
          running_ptr(run), print_mtx(pmtx), id(i), is_headless(hl)
    {
        player = new Player();
    }
    ~Env() { reset_env(); delete player; }

    void reset_env() {
        clear_container(enemies);
        clear_container(p_bullets);
        clear_container(e_bullets);
        score = 0; spawn_timer = 0; player->health = 100;
        total_reward = 0.f;
    }

    Vec get_state() const {
        return build_state(player, enemies, e_bullets);
    }

    void run() {
        Episode current_ep;
        while (running_ptr->load()) {
            if (step_count % 128 == 0) {
                std::lock_guard<std::mutex> lk(*net_mtx);
                local_net.copy_from(*shared_net);
            }

            player->update();
            Vec state = get_state();
            auto [act, val] = local_net.forward(state);
            float eps = epsilon_ptr->load();
            float aic = ai_control_ptr->load();

            // Exploration noise
            for (int i = 0; i < 4; i++) {
                act[i] += RNG::normal(0, eps);
                act[i] = std::clamp(act[i], -1.f, 1.f);
            }
            act[SHOOT_PROB] = std::clamp(act[SHOOT_PROB] + RNG::normal(0, eps*0.4f), 0.f, 1.f);

            // FIX: compute old_log_prob BEFORE blending aim (act reflects
            // the policy output, blending is a deterministic post-process)
            float old_lp = compute_log_prob(act, act, eps);

            // Algorithm blending
            Vector2 aim_dir(act[AIM_X], act[AIM_Y]);
            Enemy* near_ep = nullptr; float med = 1e9f;
            for (auto* e : enemies) {
                float d = player->pos.distance_to(e->pos);
                if (d < med) { med = d; near_ep = e; }
            }
            if (near_ep) {
                float bspd = 10.f; float t = med / bspd;
                Vector2 pred = (near_ep->pos + near_ep->vel_smooth * t - player->pos).normalize();
                aim_dir = (aim_dir.normalize() * (1.f - aic) + pred * aic).normalize();
            }
            act[AIM_X] = aim_dir.x; act[AIM_Y] = aim_dir.y;

            player->move({act[MOVE_X], act[MOVE_Y]}, true);
            float reward = 0.f;
            if (act[SHOOT_PROB] > 0.5f && player->can_shoot()) {
                p_bullets.push_back(new Bullet(player->pos, player->pos + aim_dir * 1200.f, {120, 255, 120, 255}, 10, 10.f));
                player->fire();
                total_shots_ptr->fetch_add(1);
                reward -= 0.5f; // Increased bullet cost to discourage mindless shooting
            }

            bool done = false;
            perform_sim_step(player, enemies, p_bullets, e_bullets, nullptr, score, reward, done, total_hits_ptr);
            total_reward += reward;

            Vec ns = get_state();
            current_ep.push_back({state, act, reward, val, old_lp, eps, done, ns});
            step_count++;

            // Cap episode length to prevent stale data and buffer flooding
            if (current_ep.size() >= 2000) done = true;

            if (done) {
                if (is_headless) {
                    std::lock_guard<std::mutex> lk(*print_mtx);
                    std::cout << "[Env " << id << "] Episode Done: Score=" << score 
                              << " Steps=" << current_ep.size() 
                              << " Reward=" << std::fixed << std::setprecision(2) << total_reward 
                              << std::defaultfloat << "\n";
                }
                buffer->push(std::move(current_ep));
                reset_env();
                current_ep.clear();
            }

            // FIX: integer division made explicit with cast
            int max_enemies = std::min(6, 2 + score / 500);
            if ((int)enemies.size() < max_enemies && ++spawn_timer >= 60) {
                enemies.push_back(new Enemy(player, e_bullets));
                spawn_timer = 0;
            }
            std::this_thread::yield();
        }
    }
};

// ─────────────────────────────────────────────────────────────
// Background Class
// ─────────────────────────────────────────────────────────────
class Background {
    SDL_Texture* tex = nullptr;
    int bw = 0, bh = 0;
    float scroll = 0;
public:
    explicit Background(SDL_Renderer* r) {
        tex = IMG_LoadTexture(r, "back.jpg");
        if (tex) {
            SDL_QueryTexture(tex, NULL, NULL, &bw, &bh);
            SDL_SetTextureColorMod(tex, 120, 120, 160);
        } else {
            std::cout << "Warning: back.jpg not found — using solid background.\n";
            bw = WIDTH; bh = HEIGHT;
        }
    }
    ~Background() { if (tex) SDL_DestroyTexture(tex); }
    void update() { scroll += 0.8f; if (scroll >= bw) scroll = 0; }
    void draw(SDL_Renderer* r, int ox, int oy) {
        if (!tex) {
            SDL_SetRenderDrawColor(r, 30, 30, 45, 255);
            SDL_Rect rect = {0, 0, WIDTH, HEIGHT};
            SDL_RenderFillRect(r, &rect);
            return;
        }
        for (float x = -scroll; x < WIDTH; x += static_cast<float>(bw)) {
            SDL_Rect dst = {static_cast<int>(x) + ox, oy, bw, HEIGHT};
            SDL_RenderCopy(r, tex, NULL, &dst);
        }
    }
};

// ─────────────────────────────────────────────────────────────
// Main Game Controller
// ─────────────────────────────────────────────────────────────
class Game {
    SDL_Window*   window   = nullptr;
    SDL_Renderer* renderer = nullptr;
    TTF_Font*     font     = nullptr;
    ActorCriticNet* net    = nullptr;
    Background*   bg       = nullptr;

    ReplayBuffer buffer;
    std::mutex   net_mtx, print_mtx;
    std::vector<Env*>       envs;
    std::vector<std::thread> env_threads;
    std::thread* train_thread = nullptr;

    std::atomic<bool>  running{true};
    std::atomic<float> epsilon{0.25f}, ai_control{1.0f};
    std::atomic<int>   ppo_count{0};
    std::atomic<long long> total_shots{0}, total_hits{0};

    bool ai_mode = true, headless = false;

    // UI-mode simulation state
    Player*                player = nullptr;
    std::vector<Enemy*>    enemies;
    std::vector<Bullet*>   p_bullets, e_bullets;
    std::vector<Particle*> particles;
    int score = 0, spawn_timer = 0;

    // FIX: separate local copy for the render thread; avoids data race
    // from the previous static-local pattern.
    ActorCriticNet vis_net;

public:
    ~Game() { cleanup(); }

    bool init(bool ai, bool hl) {
        ai_mode = ai; headless = hl;
        if (!headless) {
            if (SDL_Init(SDL_INIT_VIDEO) < 0) return false;
            IMG_Init(IMG_INIT_JPG); TTF_Init();
            window = SDL_CreateWindow("Spiral Shooter ── Multithreaded PPO",
                                        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                        WIDTH, HEIGHT, 0);
            renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
            const char* fts[] = {
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                nullptr
            };
            for (int i = 0; fts[i]; i++) { font = TTF_OpenFont(fts[i], 22); if (font) break; }
            bg     = new Background(renderer);
            player = new Player();
        } else {
            if (SDL_Init(0) < 0) return false;
        }

        net = new ActorCriticNet();
        if (ai_mode) {
            net->load("model.weights");
            load_params();
            train_thread = new std::thread(&Game::train_loop, this);
            for (int i = 0; i < N_ENVS; i++) {
                Env* e = new Env(i, net, &net_mtx, &buffer, &epsilon, &ai_control, &total_shots, &total_hits, &running, &print_mtx, headless);
                envs.push_back(e);
                env_threads.emplace_back(&Env::run, e);
            }
        }
        return true;
    }

    void train_loop() {
        while (running) {
            if (buffer.size() < 4) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            auto episodes = buffer.swap_out();
            std::vector<Exp> data;
            for (auto& ep : episodes) data.insert(data.end(), ep.begin(), ep.end());
            
            // Limit training to the most recent N steps to keep it fast and on-policy
            constexpr int MAX_TRAIN_STEPS = 32768;
            if (data.size() > MAX_TRAIN_STEPS) {
                data.erase(data.begin(), data.end() - MAX_TRAIN_STEPS);
            }
            
            int N = (int)data.size();
            if (N < BATCH_SIZE) continue;

            // ─────────────────────────────────────────────────
            // FIX: GAE computed per-episode, never crossing
            // boundaries between different environment episodes.
            // ─────────────────────────────────────────────────
            std::vector<float> adv(N, 0.f), ret(N, 0.f);
            {
                // Find episode boundary indices first
                // Walk through data and reset GAE at each done flag
                float gae = 0.f;
                for (int i = N - 1; i >= 0; i--) {
                    // If this step is terminal, bootstrap value = 0
                    // If next step is from a *different* episode boundary,
                    // data[i].done already handles that.
                    float next_v = data[i].done ? 0.f
                                  : (i + 1 < N ? data[i+1].v : 0.f);
                    // Reset accumulated GAE at episode boundaries
                    if (data[i].done) gae = 0.f;
                    float delta = data[i].r + GAMMA * next_v - data[i].v;
                    gae = delta + GAMMA * LAMBDA_GAE * gae;
                    adv[i] = gae;
                    ret[i] = gae + data[i].v;
                }
            }

            // Normalise advantages
            float am = 0.f;
            for (float a : adv) am += a;
            am /= static_cast<float>(N);
            float as2 = 0.f;
            for (float a : adv) as2 += (a - am) * (a - am);
            as2 = std::sqrt(as2 / static_cast<float>(N) + 1e-8f);
            for (float& a : adv) a = (a - am) / as2;

            // Build batches
            std::vector<Vec>   s_batch(N), a_batch(N);
            std::vector<float> lp_batch(N), e_batch(N);
            for (int i = 0; i < N; i++) {
                s_batch[i]  = data[i].s;
                a_batch[i]  = data[i].a;
                lp_batch[i] = data[i].old_log_prob; // FIX: pre-computed
                e_batch[i]  = data[i].eps;
            }

            { std::lock_guard<std::mutex> lk(net_mtx); net->snapshot(); }

            for (int e = 0; e < PPO_EPOCHS; e++) {
                if (!running.load()) break;
                net->ppo_update(s_batch, a_batch, adv, ret, lp_batch, e_batch, net_mtx, running);
            }

            int pc = ++ppo_count;
            if (pc % 10 == 0)  ai_control.store(std::max(0.1f, ai_control.load() * 0.995f));
            epsilon.store(std::max(0.02f, epsilon.load() * 0.9995f));

            if (pc % 50 == 0) {
                std::lock_guard<std::mutex> lk(net_mtx);
                net->save("model.weights");
                save_params();
            }
            if (headless || (pc % 5 == 0)) {
                std::cout << "[Train] PPO=" << pc
                          << " steps=" << N
                          << " aic="   << ai_control.load()
                          << " eps="   << epsilon.load()
                          << '\n';
            }
        }
    }

    void run() {
        if (headless) {
            while (running) std::this_thread::sleep_for(std::chrono::seconds(1));
            return;
        }

        while (running) {
            handle_events();
            update_ui_sim();
            render();
            SDL_Delay(16);
        }
    }

private:
    void load_params() {
        std::ifstream fe("epsilon.txt"); if (fe) { float f; fe >> f; epsilon.store(f); }
        std::ifstream fa("ai_control.txt"); if (fa) { float f; fa >> f; ai_control.store(f); }
        std::ifstream fp("generation.txt"); if (fp) { int i; fp >> i; ppo_count.store(i); }
        std::ifstream fs("stats.txt"); if (fs) { long long s, h; fs >> s >> h; total_shots.store(s); total_hits.store(h); }
    }

    void save_params() {
        std::ofstream fe("epsilon.txt"); if (fe) fe << epsilon.load();
        std::ofstream fa("ai_control.txt"); if (fa) fa << ai_control.load();
        std::ofstream fp("generation.txt"); if (fp) fp << ppo_count.load();
        std::ofstream fs("stats.txt"); if (fs) fs << total_shots.load() << " " << total_hits.load();
    }

    void handle_events() {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) running = false;
            if (e.type == SDL_KEYDOWN && e.key.keysym.scancode == SDL_SCANCODE_ESCAPE)
                running = false;
        }
    }

    void update_ui_sim() {
        player->update();
        if (bg) bg->update();

        Vec state = build_state(player, enemies, e_bullets);
        Vec act(NUM_ACTIONS, 0.f);

        // FIX: use a named member vis_net instead of a static local to
        // avoid data races when try_lock fails on the first frame.
        if (net_mtx.try_lock()) {
            vis_net.copy_from(*net);
            net_mtx.unlock();
        }
        auto [a, v] = vis_net.forward(state);
        act = a;

        float aic = ai_control.load();
        Vector2 aim_dir(act[AIM_X], act[AIM_Y]);

        Enemy* near_ep = nullptr; float min_ed = 1e9f;
        for (auto* e : enemies) {
            float d = player->pos.distance_to(e->pos);
            if (d < min_ed) { min_ed = d; near_ep = e; }
        }
        if (near_ep) {
            float bspd = 10.f; float t = min_ed / bspd;
            Vector2 pred = (near_ep->pos + near_ep->vel_smooth * t - player->pos).normalize();
            aim_dir = (aim_dir.normalize() * (1.f - aic) + pred * aic).normalize();
        }

        player->move({act[MOVE_X], act[MOVE_Y]}, true);
        if (act[SHOOT_PROB] > 0.5f && player->can_shoot() && near_ep) {
            p_bullets.push_back(new Bullet(player->pos, player->pos + aim_dir * 1200.f, {120, 255, 120, 255}, 10, 10.f));
            player->fire();
            total_shots.fetch_add(1);
        }

        bool done = false; float dummy_r = 0.f;
        perform_sim_step(player, enemies, p_bullets, e_bullets, &particles, score, dummy_r, done, &total_hits);
        if (done) {
            clear_container(enemies);
            clear_container(p_bullets);
            clear_container(e_bullets);
            score = 0; player->health = 100;
        }

        int max_enemies = std::min(6, 2 + score / 500);
        if ((int)enemies.size() < max_enemies && ++spawn_timer >= 60) {
            enemies.push_back(new Enemy(player, e_bullets));
            spawn_timer = 0;
        }
    }

    void render() {
        SDL_SetRenderDrawColor(renderer, 20, 20, 28, 255);
        SDL_RenderClear(renderer);
        if (bg) bg->draw(renderer, 0, 0);

        for (auto* p : particles) p->draw(renderer, 0, 0);
        for (auto* b : p_bullets) b->draw(renderer, 0, 0);
        for (auto* b : e_bullets) b->draw(renderer, 0, 0);
        for (auto* e : enemies) {
            e->draw(renderer, 0, 0);
            char hbuf[16];
            snprintf(hbuf, sizeof(hbuf), "%d", e->health); // FIX: snprintf
            render_text(hbuf, (int)e->pos.x, (int)e->pos.y - 30, {255, 100, 100, 255}, true);
        }
        player->draw(renderer, 0, 0);
        draw_hud();
        SDL_RenderPresent(renderer);
    }

    void draw_hud() {
        char buf[256];
        float acc = total_shots.load() > 0 ? (float)total_hits.load() / total_shots.load() * 100.f : 0.f;

        // Line 1: Training Stats
        snprintf(buf, sizeof(buf), "PPO Iter: %d | EPS: %.3f | AIC: %.3f",
                 ppo_count.load(), epsilon.load(), ai_control.load());
        render_text(buf, 20, 20, {255, 255, 255, 255});

        // Line 2: Performance Stats
        snprintf(buf, sizeof(buf), "Accuracy: %.1f%% | Shots: %lld | Hits: %lld",
                 acc, total_shots.load(), total_hits.load());
        render_text(buf, 20, 50, {200, 255, 200, 255});

        // Line 3: Current Session
        snprintf(buf, sizeof(buf), "Score: %d | HP: %d", score, player->health);
        render_text(buf, 20, 80, (player->health > 30 ? SDL_Color{100,255,255,255} : SDL_Color{255,50,50,255}));
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

    void cleanup() {
        running = false;
        if (font)     { TTF_CloseFont(font);          font     = nullptr; }
        if (renderer) { SDL_DestroyRenderer(renderer); renderer = nullptr; }
        if (window)   { SDL_DestroyWindow(window);     window   = nullptr; }
        TTF_Quit(); IMG_Quit(); SDL_Quit();

        for (auto& t : env_threads) if (t.joinable()) t.join();
        env_threads.clear();
        if (train_thread) {
            if (train_thread->joinable()) train_thread->join();
            delete train_thread; train_thread = nullptr;
        }
        clear_container(envs);
        if (net) {
            net->save("model.weights");
            save_params();
            delete net; net = nullptr;
        }
        if (bg)  { delete bg; bg = nullptr; }
        if (player) { delete player; player = nullptr; }
        clear_container(enemies);
        clear_container(p_bullets);
        clear_container(e_bullets);
        clear_container(particles);
    }
};

// ─────────────────────────────────────────────────────────────
// Static assert: ensure atomic<float> operations are sensible.
// On most x86/ARM platforms this is lock-free; warn at compile time
// if not, so the developer knows to audit performance.
// ─────────────────────────────────────────────────────────────
static_assert(sizeof(std::atomic<float>) == sizeof(float),
              "std::atomic<float> has unexpected size — check lock-free support.");

int main(int argc, char* argv[]) {
    bool ai = true, headless = false;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "h" || arg == "human")           ai       = false;
        if (arg == "headless" || arg == "--headless") headless = true;
    }
    Game game;
    if (game.init(ai, headless)) game.run();
    return 0;
}
