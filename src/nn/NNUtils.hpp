#pragma once
#include "../common/Types.hpp"
#include <cmath>
#include <vector>

namespace nn {
    inline Mat make_mat(int rows, int cols, float val = 0.f) {
        return Mat(rows, Vec(cols, val));
    }

    inline Vec mat_vec_mul(const Mat& W, const Vec& x) {
        Vec out(W.size(), 0.f);
        for (int i = 0; i < (int)W.size(); i++)
            for (int j = 0; j < (int)x.size(); j++)
                out[i] += W[i][j] * x[j];
        return out;
    }

    inline void add_bias(Vec& v, const Vec& b) {
        for (int i = 0; i < (int)v.size(); i++) v[i] += b[i];
    }

    inline Vec leaky_relu(const Vec& v, float alpha = 0.01f) {
        Vec out(v.size());
        for (int i = 0; i < (int)v.size(); i++)
            out[i] = v[i] >= 0.f ? v[i] : alpha * v[i];
        return out;
    }

    inline Vec d_leaky_relu(const Vec& pre, float alpha = 0.01f) {
        Vec out(pre.size());
        for (int i = 0; i < (int)pre.size(); i++)
            out[i] = pre[i] >= 0.f ? 1.f : alpha;
        return out;
    }

    inline void layer_norm(Vec& v, float eps = 1e-5f) {
        float mean = 0, var = 0;
        for (float x : v) mean += x;
        mean /= v.size();
        for (float x : v) var += (x - mean) * (x - mean);
        var /= v.size();
        float inv = 1.f / std::sqrt(var + eps);
        for (float& x : v) x = (x - mean) * inv;
    }

    inline void softmax(Vec& v, int start, int count) {
        float max_val = -1e9f;
        for (int i = 0; i < count; i++) max_val = std::max(max_val, v[start + i]);
        float sum = 0;
        for (int i = 0; i < count; i++) {
            v[start + i] = std::exp(v[start + i] - max_val);
            sum += v[start + i];
        }
        for (int i = 0; i < count; i++) v[start + i] /= sum;
    }
}
