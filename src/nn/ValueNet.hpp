#pragma once
#include "../common/Types.hpp"
#include "../common/Constants.hpp"
#include "Optimizer.hpp"
#include <fstream>

namespace nn {
    struct ValueNet {
        Mat vw1, vw2, vw3, vw4;
        Vec vb1, vb2, vb3, vb4;
        AdamState vas1, vas2, vas3, vas4;
        float lr = 1e-3f;

        static constexpr int VH1 = 128, VH2 = 128, VH3 = 64;

        ValueNet();

        struct VCache { Vec h1, h2, h3, pre1, pre2, pre3; };

        Vec forward(const Vec& s, VCache* cache = nullptr) const;
        void train_step(const Vec& s, const Vec& target);

        void save(std::ofstream& f) const;
        void load(std::ifstream& f);
    };
}
