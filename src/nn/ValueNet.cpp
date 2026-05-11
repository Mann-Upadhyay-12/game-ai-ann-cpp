#include "ValueNet.hpp"
#include "../common/RNG.hpp"
#include "NNUtils.hpp"
#include <iostream>

namespace nn {
    ValueNet::ValueNet() {
        auto init_mat = [&](Mat& W, Vec& b, int rows, int cols) {
            float s = std::sqrt(2.f / cols);
            W = make_mat(rows, cols);
            b.assign(rows, 0.f);
            for (auto& row : W) for (auto& w : row) w = RNG::normal(0, s);
        };
        init_mat(vw1, vb1, VH1, constants::STATE_DIM);
        init_mat(vw2, vb2, VH2, VH1);
        init_mat(vw3, vb3, VH3, VH2);
        init_mat(vw4, vb4, 2,   VH3);
        vas1 = AdamState(VH1, constants::STATE_DIM);
        vas2 = AdamState(VH2, VH1);
        vas3 = AdamState(VH3, VH2);
        vas4 = AdamState(2,   VH3);
    }

    Vec ValueNet::forward(const Vec& s, VCache* cache) const {
        Vec pre1 = mat_vec_mul(vw1, s); add_bias(pre1, vb1);
        Vec h1   = leaky_relu(pre1);    layer_norm(h1);
        Vec pre2 = mat_vec_mul(vw2, h1); add_bias(pre2, vb2);
        Vec h2   = leaky_relu(pre2);    layer_norm(h2);
        Vec pre3 = mat_vec_mul(vw3, h2); add_bias(pre3, vb3);
        Vec h3   = leaky_relu(pre3);    layer_norm(h3);
        Vec out  = mat_vec_mul(vw4, h3); add_bias(out, vb4);
        if (cache) *cache = {h1, h2, h3, pre1, pre2, pre3};
        return out;
    }

    void ValueNet::train_step(const Vec& s, const Vec& target) {
        VCache cache;
        Vec pred = forward(s, &cache);
        Vec dout(2);
        for (int i = 0; i < 2; i++) dout[i] = target[i] - pred[i];

        Mat dvw4 = make_mat(2, VH3);
        Vec dvb4(2, 0.f);
        Vec h3_err(VH3, 0.f);
        for (int i = 0; i < VH3; i++) {
            float e = 0;
            for (int j = 0; j < 2; j++) {
                dvw4[j][i] = dout[j] * cache.h3[i];
                e         += vw4[j][i] * dout[j];
            }
            h3_err[i] = e;
        }
        for (int j = 0; j < 2; j++) dvb4[j] = dout[j];

        Vec d3 = d_leaky_relu(cache.pre3);
        Mat dvw3 = make_mat(VH3, VH2);
        Vec dvb3(VH3, 0.f);
        Vec h2_err(VH2, 0.f);
        for (int i = 0; i < VH2; i++) {
            float e = 0;
            for (int j = 0; j < VH3; j++) {
                float g = h3_err[j] * d3[j];
                dvw3[j][i] += g * cache.h2[i];
                e          += vw3[j][i] * g;
            }
            h2_err[i] = e;
        }
        for (int j = 0; j < VH3; j++) dvb3[j] = h3_err[j] * d3[j];

        Vec d2 = d_leaky_relu(cache.pre2);
        Mat dvw2 = make_mat(VH2, VH1);
        Vec dvb2(VH2, 0.f);
        Vec h1_err(VH1, 0.f);
        for (int i = 0; i < VH1; i++) {
            float e = 0;
            for (int j = 0; j < VH2; j++) {
                float g = h2_err[j] * d2[j];
                dvw2[j][i] += g * cache.h1[i];
                e          += vw2[j][i] * g;
            }
            h1_err[i] = e;
        }
        for (int j = 0; j < VH2; j++) dvb2[j] = h2_err[j] * d2[j];

        Vec d1 = d_leaky_relu(cache.pre1);
        Mat dvw1 = make_mat(VH1, constants::STATE_DIM);
        Vec dvb1(VH1, 0.f);
        for (int i = 0; i < constants::STATE_DIM; i++)
            for (int j = 0; j < VH1; j++)
                dvw1[j][i] = h1_err[j] * d1[j] * s[i];
        for (int j = 0; j < VH1; j++) dvb1[j] = h1_err[j] * d1[j];

        adam_update(vw1, vb1, dvw1, dvb1, vas1, lr);
        adam_update(vw2, vb2, dvw2, dvb2, vas2, lr);
        adam_update(vw3, vb3, dvw3, dvb3, vas3, lr);
        adam_update(vw4, vb4, dvw4, dvb4, vas4, lr);
    }

    void ValueNet::save(std::ofstream& f) const {
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
        wmat(vw1); wmat(vw2); wmat(vw3); wmat(vw4);
        wvec(vb1); wvec(vb2); wvec(vb3); wvec(vb4);
        wadam(vas1); wadam(vas2); wadam(vas3); wadam(vas4);
    }

    void ValueNet::load(std::ifstream& f) {
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
        rmat(vw1); rmat(vw2); rmat(vw3); rmat(vw4);
        rvec(vb1); rvec(vb2); rvec(vb3); rvec(vb4);
        radam(vas1); radam(vas2); radam(vas3); radam(vas4);
    }
}
