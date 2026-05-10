#pragma once
#include "Entities.hpp"
#include "../nn/PolicyNet.hpp"
#include <vector>
#include <string>

namespace game {
    class GameSim {
    public:
        LogicPlayer*               player   = nullptr;
        std::vector<LogicEnemy*>   enemies;
        std::vector<LogicBullet*>  p_bullets, e_bullets;

        nn::PolicyNet*   net    = nullptr;

        struct StepData {
            Vec   state, action;
            float reward;
            Vec   next_state;
            float log_prob;
            bool  done;
        };
        std::vector<StepData> rollout;

        int   score        = 0;
        float total_reward = 0.f;
        int   spawn_timer  = 0;
        bool  alive        = true;

        float epsilon      = 0.3f;
        static constexpr float EPS_MIN   = 0.08f;
        static constexpr float EPS_DECAY = 0.99995f;

        int   train_every = 128;
        int   step_count  = 0;

        int shots_fired = 0, shots_hit = 0;

        int   total_episodes = 0;
        float best_score     = 0;

        std::vector<Vec> state_history;

        GameSim(nn::PolicyNet* n);

        ~GameSim();

        void clear_all();
        void finish_episode();
        void reset();
        void update_text_files();
        bool is_visible(Vector2 p) const;
        Vec get_single_state() const;
        Vec get_state() const;
        Vector2 predict_aim(const Vector2& p_pos, const Vector2& e_pos,
                            const Vector2& e_vel, float b_speed);
        bool step();

    private:
        bool update_entities(float& reward);
    };
}
