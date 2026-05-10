#include "nn/PolicyNet.hpp"
#include "game/GameSim.hpp"
#include <iostream>
#include <string>
#include <chrono>

#ifndef HEADLESS
#include "rendering/VisualGame.hpp"
#endif

void run_headless(nn::PolicyNet* net, int steps, float eps, int episodes, float best) {
    std::cout << "[Headless] A2C training. Ctrl+C to stop.\n";
    game::GameSim sim(net);
    sim.step_count     = steps;
    sim.epsilon        = eps;
    sim.total_episodes = episodes;
    sim.best_score     = best;

    auto t_start = std::chrono::steady_clock::now();
    long long total_steps = 0;

    while (true) {
        bool alive = sim.step();
        total_steps++;
        if (!alive) sim.reset();
        if (total_steps % 10000 == 0) {
            auto now    = std::chrono::steady_clock::now();
            double secs = std::chrono::duration<double>(now - t_start).count();
            std::cout << "[Headless] steps=" << total_steps
                      << " sps=" << (int)(total_steps/secs)
                      << " eps=" << sim.epsilon
                      << "\n";
        }
    }
}

int main(int argc, char* argv[]) {
    std::string mode = (argc > 1) ? std::string(argv[1]) : "ai";
    nn::PolicyNet net;
    
    // Load attempts
    auto d_latest = net.load("latest.weights");
    auto d_best   = net.load("best.weights");
    auto d_model  = net.load("model.weights");

    nn::PolicyNet::ModelData data = d_latest;
    std::string best_file = "latest.weights";

    if (d_best.steps > data.steps) {
        data = d_best;
        best_file = "best.weights";
    }
    if (d_model.steps > data.steps) {
        data = d_model;
        best_file = "model.weights";
    }

    // Final load of the best one found
    data = net.load(best_file);
    std::cout << "[Model] Resuming from " << best_file << " (" << data.steps << " steps)\n";

#ifdef HEADLESS
    (void)mode;
    run_headless(&net, data.steps, data.eps, data.episodes, data.best);
#else
    if (mode == "headless" || mode == "fast") {
        run_headless(&net, data.steps, data.eps, data.episodes, data.best);
    } else {
        bool ai = !(mode == "h" || mode == "human");
        rendering::VisualGame game;
        if (!game.init(ai, &net, data.steps, data.eps, data.episodes, data.best)) return 1;
        game.run();
    }
#endif
    return 0;
}
