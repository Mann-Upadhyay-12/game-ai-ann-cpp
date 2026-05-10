#include "PolicyNet.hpp"
#include "../common/RNG.hpp"
#include "NNUtils.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>

namespace nn {
    PolicyNet::PolicyNet() {
        auto he = [](int fan_in) {
            return RNG::normal(0, std::sqrt(2.f / fan_in));
        };
        auto init_mat = [&](Mat& W, Vec& b, int rows, int cols) {
            W = make_mat(rows, cols);
            b.assign(rows, 0.f);
            for (auto& row : W) for (auto& w : row) w = he(cols);
        };
        init_mat(w1, b1, H1, constants::STATE_DIM);
        init_mat(w2, b2, H2, H1);
        init_mat(w3, b3, H3, H2);
        init_mat(w4, b4, constants::NUM_ACTIONS, H3);
        
        as1 = AdamState(H1, constants::STATE_DIM);
        as2 = AdamState(H2, H1);
        as3 = AdamState(H3, H2);
        as4 = AdamState(constants::NUM_ACTIONS, H3);

        lr = 5e-5f; // User Priority: Low LR for stability
    }

    Vec PolicyNet::forward(const Vec& s, Cache* cache) const {
        Vec pre1 = mat_vec_mul(w1, s); add_bias(pre1, b1);
        Vec h1   = leaky_relu(pre1);   layer_norm(h1);
        Vec pre2 = mat_vec_mul(w2, h1); add_bias(pre2, b2);
        Vec h2   = leaky_relu(pre2);    layer_norm(h2);
        Vec pre3 = mat_vec_mul(w3, h2); add_bias(pre3, b3);
        Vec h3   = leaky_relu(pre3);    layer_norm(h3);
        Vec out  = mat_vec_mul(w4, h3); add_bias(out, b4);

        // Discretized Heads (D)
        softmax(out, 0, constants::MOVE_ACTIONS);
        softmax(out, constants::MOVE_ACTIONS, constants::SHOOT_ACTIONS);

        if (cache) *cache = {h1, h2, h3, pre1, pre2, pre3};
        return out;
    }

    void PolicyNet::train_ppo(const std::vector<Vec>& states,
                              const std::vector<Vec>& actions_taken,
                              const std::vector<float>& returns,
                              const std::vector<float>& old_log_probs,
                              float clip_eps,
                              int ppo_epochs)
    {
        int N = (int)states.size();
        if (N == 0) return;

        // Advantage calculation
        std::vector<float> advantages(N);
        for (int i = 0; i < N; i++) {
            float V = critic.forward(states[i]);
            advantages[i] = returns[i] - V;
            critic.train_step(states[i], returns[i]);
        }

        float adv_mean = 0;
        for (float a : advantages) adv_mean += a;
        adv_mean /= N;
        float adv_std = 0;
        for (float a : advantages) adv_std += (a - adv_mean) * (a - adv_mean);
        adv_std = std::sqrt(adv_std / N + 1e-8f);
        for (float& a : advantages) a = (a - adv_mean) / adv_std;

        for (int epoch = 0; epoch < ppo_epochs; epoch++) {
            Mat dw1 = make_mat(H1, constants::STATE_DIM), dw2 = make_mat(H2, H1),
                dw3 = make_mat(H3, H2),        dw4 = make_mat(constants::NUM_ACTIONS, H3);
            Vec db1(H1,0), db2(H2,0), db3(H3,0), db4(constants::NUM_ACTIONS,0);

            for (int idx = 0; idx < N; idx++) {
                Cache cache;
                Vec pred = forward(states[idx], &cache);

                int move_a  = (int)actions_taken[idx][0];
                int shoot_a = (int)actions_taken[idx][1];

                float cur_lp = std::log(pred[move_a] + 1e-9f) +
                               std::log(pred[constants::MOVE_ACTIONS + shoot_a] + 1e-9f);

                float ratio = std::exp(cur_lp - old_log_probs[idx]);
                float A = advantages[idx];

                // Standard PPO Gradient scale: ratio * A (clipped)
                float pg_scale = ratio * A;
                if (A >= 0) {
                    if (ratio > 1.0f + clip_eps) pg_scale = 0.f;
                } else {
                    if (ratio < 1.0f - clip_eps) pg_scale = 0.f;
                }

                Vec out_err(constants::NUM_ACTIONS, 0.f);
                auto add_head_loss = [&](int start, int count, int action_taken, float ent_coeff) {
                    for (int i = 0; i < count; i++) {
                        float target = (i == action_taken) ? 1.f : 0.f;
                        // Policy gradient
                        out_err[start + i] += pg_scale * (target - pred[start + i]);
                        // Entropy bonus: encourages exploration
                        float p = pred[start + i];
                        out_err[start + i] += ent_coeff * p * (-std::log(p + 1e-9f));
                    }
                };

                add_head_loss(0, constants::MOVE_ACTIONS, move_a, 0.03f);
                add_head_loss(constants::MOVE_ACTIONS, constants::SHOOT_ACTIONS, shoot_a, 0.05f);

                Vec h3_err(H3, 0.f);
                for (int i = 0; i < H3; i++) {
                    float e = 0;
                    for (int j = 0; j < constants::NUM_ACTIONS; j++) {
                        dw4[j][i] += out_err[j] * cache.h3[i];
                        e         += w4[j][i] * out_err[j];
                    }
                    h3_err[i] = e;
                }
                for (int j = 0; j < constants::NUM_ACTIONS; j++) db4[j] += out_err[j];

                Vec d3 = d_leaky_relu(cache.pre3);
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

                Vec d2 = d_leaky_relu(cache.pre2);
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

                Vec d1 = d_leaky_relu(cache.pre1);
                for (int i = 0; i < constants::STATE_DIM; i++)
                    for (int j = 0; j < H1; j++)
                        dw1[j][i] += h1_err[j] * d1[j] * states[idx][i];
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
    }

    void PolicyNet::save(const std::string& path, int steps, float eps, int episodes, float best) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) { std::cerr << "Save failed: " << path << "\n"; return; }
        auto wmat = [&](const Mat& M) {
            for (auto& r : M) f.write((const char*)r.data(), r.size()*sizeof(float));
        };
        auto wvec = [&](const Vec& v) {
            f.write((const char*)v.data(), v.size()*sizeof(float));
        };
        auto wadam = [&](const AdamState& st) {
            wmat(st.m); wmat(st.v);
            wvec(st.mb); wvec(st.vb);
            f.write((const char*)&st.t, sizeof(int));
        };

        wmat(w1); wmat(w2); wmat(w3); wmat(w4);
        wvec(b1); wvec(b2); wvec(b3); wvec(b4);
        wadam(as1); wadam(as2); wadam(as3); wadam(as4);

        critic.save(f);

        f.write((const char*)&steps,    sizeof(int));
        f.write((const char*)&eps,      sizeof(float));
        f.write((const char*)&episodes, sizeof(int));
        f.write((const char*)&best,     sizeof(float));
        std::cout << "[Model] Saved (steps=" << steps << " eps=" << eps
                  << " ep=" << episodes << " best=" << (int)best << ")\n";
    }

    PolicyNet::ModelData PolicyNet::load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) { std::cout << "[Model] No weights found, starting fresh.\n"; return {0, 0.3f, 0, 0}; }
        auto rmat = [&](Mat& M) {
            for (auto& r : M) f.read((char*)r.data(), r.size()*sizeof(float));
        };
        auto rvec = [&](Vec& v) {
            f.read((char*)v.data(), v.size()*sizeof(float));
        };
        auto radam = [&](AdamState& st) {
            rmat(st.m); rmat(st.v);
            rvec(st.mb); rvec(st.vb);
            f.read((char*)&st.t, sizeof(int));
        };

        rmat(w1); rmat(w2); rmat(w3); rmat(w4);
        rvec(b1); rvec(b2); rvec(b3); rvec(b4);
        radam(as1); radam(as2); radam(as3); radam(as4);

        critic.load(f);

        int   steps = 0, episodes = 0;
        float eps = 0.3f, best = 0.f;
        f.read((char*)&steps,    sizeof(int));
        f.read((char*)&eps,      sizeof(float));
        f.read((char*)&episodes, sizeof(int));
        f.read((char*)&best,     sizeof(float));
        std::cout << "[Model] Loaded (steps=" << steps << " eps=" << eps
                  << " ep=" << episodes << " best=" << (int)best << ")\n";
        return {steps, eps, episodes, best};
    }
}
