#pragma once
#include <random>

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
    static float normal(float mean = 0.f, float std = 1.f) {
        return std::normal_distribution<float>(mean, std)(gen());
    }
    static int categorical(const std::vector<float>& probs, int start, int count) {
        float r = flt(0, 1);
        float sum = 0;
        for (int i = 0; i < count; i++) {
            sum += probs[start + i];
            if (r <= sum) return i;
        }
        return count - 1;
    }
    static int argmax(const std::vector<float>& probs, int start, int count) {
        int best = 0;
        float max_p = -1e9f;
        for (int i = 0; i < count; i++) {
            if (probs[start + i] > max_p) {
                max_p = probs[start + i];
                best = i;
            }
        }
        return best;
    }
};
