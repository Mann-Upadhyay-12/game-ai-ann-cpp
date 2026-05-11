#pragma once
#include "../common/Types.hpp"
#include "../common/Constants.hpp"
#include "Optimizer.hpp"
#include "ValueNet.hpp"
#include <fstream>
#include <string>
#include <vector>

namespace nn {
    struct PolicyNet {
        Mat w1, w2, w3, w4;
        Vec b1, b2, b3, b4;
        AdamState as1, as2, as3, as4;
        float lr = 3e-4f;

        ValueNet critic;

        static constexpr int H1 = 128, H2 = 128, H3 = 64;

        PolicyNet();

        struct Cache { Vec h1, h2, h3, pre1, pre2, pre3; };

        Vec forward(const Vec& s, Cache* cache = nullptr) const;
        void train_ppo(const std::vector<Vec>& states,
                       const std::vector<Vec>& actions_taken,
                       const std::vector<float>& rewards_s,
                       const std::vector<float>& rewards_o,
                       const std::vector<Vec>& next_states,
                       const std::vector<bool>& dones,
                       const std::vector<float>& old_log_probs,
                       float clip_eps = 0.2f,
                       int ppo_epochs = 4);


        void save(const std::string& path, int steps, float eps, int episodes, float best) const;
        struct ModelData { int steps; float eps; int episodes; float best; };
        ModelData load(const std::string& path);
    };
}
