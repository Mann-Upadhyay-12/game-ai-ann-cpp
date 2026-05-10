#pragma once
#include "../common/Types.hpp"
#include "NNUtils.hpp"
#include <cmath>

namespace nn {
    struct AdamState {
        Mat m, v;
        Vec mb, vb;
        int t = 0;
        AdamState() = default;
        AdamState(int rows, int cols)
            : m(make_mat(rows, cols)), v(make_mat(rows, cols)),
              mb(rows, 0.f), vb(rows, 0.f) {}
    };

    inline void adam_update(Mat& W, Vec& b,
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
}
