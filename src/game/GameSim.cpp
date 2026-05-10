#include "GameSim.hpp"
#include "../common/RNG.hpp"
#include "../common/Constants.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cassert>

namespace game {
    GameSim::GameSim(nn::PolicyNet* n) : net(n) {
        player = new LogicPlayer();
        state_history.resize(constants::FRAME_STACK, Vec(constants::SINGLE_STATE_DIM, 0.f));
        epsilon = 0.3f; // Initial noise
    }
    GameSim::~GameSim() { clear_all(); delete player; }

    void GameSim::clear_all() {
        for (auto* e : enemies)   delete e;
        for (auto* b : p_bullets) delete b;
        for (auto* b : e_bullets) delete b;
        enemies.clear(); p_bullets.clear(); e_bullets.clear();
    }

    void GameSim::finish_episode() {
        if (rollout.empty()) return;
        float gamma = 0.99f;
        float G = 0.f;
        if (!rollout.back().done) {
            G = net->critic.forward(rollout.back().next_state);
        }

        std::vector<Vec> states, actions;
        std::vector<float> returns, log_probs;
        states.reserve(rollout.size());
        actions.reserve(rollout.size());
        returns.reserve(rollout.size());
        log_probs.reserve(rollout.size());

        for (int i = (int)rollout.size()-1; i >= 0; i--) {
            G = rollout[i].reward + gamma * G;
            states.push_back(rollout[i].state);
            actions.push_back(rollout[i].action);
            returns.push_back(G);
            log_probs.push_back(rollout[i].log_prob);
        }
        
        std::reverse(states.begin(), states.end());
        std::reverse(actions.begin(), actions.end());
        std::reverse(returns.begin(), returns.end());
        std::reverse(log_probs.begin(), log_probs.end());

        net->train_ppo(states, actions, returns, log_probs);
        rollout.clear();
    }

    void GameSim::reset() {
        float acc = shots_fired > 0 ? 100.f * shots_hit / shots_fired : 0.f;
        if (score > best_score) {
            best_score = (float)score;
            net->save("best.weights", step_count, epsilon, total_episodes, best_score);
        }
        total_episodes++;

        std::cout << "[Stats] ep=" << total_episodes
                  << " score=" << score
                  << " best=" << (int)best_score
                  << " acc=" << acc << "%"
                  << " (" << shots_hit << "/" << shots_fired << ")"
                  << " steps=" << step_count << "\n";

        finish_episode();
        net->save("latest.weights", step_count, epsilon, total_episodes, best_score);
        update_text_files();

        clear_all();
        delete player; player = new LogicPlayer();

        state_history.assign(constants::FRAME_STACK, Vec(constants::SINGLE_STATE_DIM, 0.f));
        score = 0; total_reward = 0; spawn_timer = 0;
        prev_min_b_dist = 1e9f; zone_edge_timer = 0;
        shots_fired = shots_hit = 0;
        alive = true;
    }

    void GameSim::update_text_files() {
        { std::ofstream f("epsilon.txt"); f << epsilon; }
        { std::ofstream f("eps.txt"); f << epsilon; }
        {
            float ai_w = std::min(1.0f, (float)step_count / 1000000.f);
            std::ofstream f("ai_control.txt"); f << ai_w;
        }
        { std::ofstream f("generation.txt"); f << total_episodes; }
        {
            std::ofstream f("stats.txt");
            f << "Score: " << score << "\n"
              << "Best: " << (int)best_score << "\n"
              << "Episodes: " << total_episodes << "\n"
              << "Steps: " << step_count << "\n";
        }
    }

    bool GameSim::is_visible(Vector2 p) const {
        return (p.x >= 0 && p.x <= constants::WIDTH && p.y >= 0 && p.y <= constants::HEIGHT);
    }

    Vec GameSim::get_single_state() const {
        Vec s;
        s.push_back(player->get_pos().x / constants::WIDTH);
        s.push_back(player->get_pos().y / constants::HEIGHT);
        s.push_back(player->get_vel().x / 8.f);
        s.push_back(player->get_vel().y / 8.f);

        float   min_ed = 1e9f;
        Vector2 near_e = {(float)constants::WIDTH/2, (float)constants::HEIGHT/2};
        LogicEnemy* near_ep = nullptr;
        bool found_enemy = false;
        for (auto* e : enemies) {
            if (!is_visible(e->get_pos())) continue;
            float d = player->get_pos().distance_to(e->get_pos());
            if (d < min_ed) { min_ed = d; near_e = e->get_pos(); near_ep = e; found_enemy = true; }
        }
        if (found_enemy) {
            Vector2 e_rel = (near_e - player->get_pos()).normalize();
            s.push_back(e_rel.x);
            s.push_back(e_rel.y);
            s.push_back(std::min(1.f, min_ed / 900.f));
            Vector2 ev = near_ep->get_vel_smooth();
            s.push_back(ev.x / 5.f);
            s.push_back(ev.y / 5.f);
        } else {
            s.push_back(0.f); s.push_back(0.f);
            s.push_back(1.f);
            s.push_back(0.f); s.push_back(0.f);
        }

        struct BulletInfo { float dist; Vector2 pos, vel; };
        std::vector<BulletInfo> found_bullets;
        for (auto* b : e_bullets) {
            if (!is_visible(b->get_pos())) continue;
            float d = player->get_pos().distance_to(b->get_pos());
            found_bullets.push_back({d, b->get_pos(), b->get_vel()});
        }
        std::sort(found_bullets.begin(), found_bullets.end(), [](const BulletInfo& a, const BulletInfo& b) {
            return a.dist < b.dist;
        });

        for (int i = 0; i < 2; i++) {
            if (i < (int)found_bullets.size()) {
                Vector2 to_p = (player->get_pos() - found_bullets[i].pos).normalize();
                Vector2 bvn  = found_bullets[i].vel.normalize();
                float threat = bvn.dot(to_p);

                Vector2 b_dir = (found_bullets[i].pos - player->get_pos()).normalize();
                s.push_back(b_dir.x);
                s.push_back(b_dir.y);
                s.push_back(std::min(1.f, found_bullets[i].dist / 500.f));
                s.push_back(bvn.x);
                s.push_back(bvn.y);
                s.push_back(threat);
            } else {
                s.push_back(0.f); s.push_back(0.f);
                s.push_back(1.f);
                s.push_back(0.f); s.push_back(0.f);
                s.push_back(-1.f);
            }
        }

        s.push_back(1.f - player->get_cd() / 10.f);
        s.push_back(player->get_health() / 100.f);
        assert((int)s.size() == constants::SINGLE_STATE_DIM);
        return s;
    }

    Vec GameSim::get_state() const {
        Vec stacked;
        stacked.reserve(constants::STATE_DIM);
        for (const auto& s : state_history) {
            stacked.insert(stacked.end(), s.begin(), s.end());
        }
        assert((int)stacked.size() == constants::STATE_DIM);
        return stacked;
    }

    Vector2 GameSim::predict_aim(const Vector2& p_pos, const Vector2& e_pos,
                        const Vector2& e_vel, float b_speed)
    {
        float   dist = p_pos.distance_to(e_pos);
        Vector2 pred = e_pos;
        for (int i = 0; i < 5; i++) {
            float t = p_pos.distance_to(pred) / b_speed;
            pred    = e_pos + e_vel * t;
        }
        if (p_pos.distance_to(pred) > dist + 250.f) pred = e_pos;
        return pred;
    }

    bool GameSim::step() {
        player->update();

        // Update state history with the latest frame
        Vec single = get_single_state();
        state_history.erase(state_history.begin());
        state_history.push_back(single);

        Vec state = get_state();
        Vec probs = net->forward(state);

        // Discrete Action Sampling - Use categorical for training
        int move_idx  = RNG::categorical(probs, 0, constants::MOVE_ACTIONS);
        int shoot_idx = RNG::categorical(probs, constants::MOVE_ACTIONS, constants::SHOOT_ACTIONS);

        // Interpretation
        static const Vector2 dirs[] = {
            {0,-1}, {0.707f,-0.707f}, {1,0}, {0.707f,0.707f},
            {0,1}, {-0.707f,0.707f}, {-1,0}, {-0.707f,-0.707f}, {0,0}
        };
        Vector2 move_dir = dirs[move_idx];
        player->move(move_dir, true);

        float reward = 0.f; // Drop survival reward — it's teaching passivity
        Vector2 p_pos = player->get_pos();

        // --- Movement rewards ---
        // Penalize staying still
        if (move_idx == 8) {
            reward -= 0.05f;
        }

        // Reward closing distance to nearest enemy (approach bonus)
        float   min_ed = 1e9f;
        Vector2 near_ep = {0,0}, near_ev = {0,0};
        bool    found_enemy = false;
        for (auto* e : enemies) {
            if (!is_visible(e->get_pos())) continue;
            float d = p_pos.distance_to(e->get_pos());
            if (d < min_ed) { min_ed = d; near_ep = e->get_pos(); near_ev = e->get_vel_smooth(); found_enemy = true; }
        }

        if (found_enemy) {
            // Reward being in engagement range (100–350px), penalize camping far away
            if (min_ed < 120.f)       reward -= 0.1f;  // too close, getting rammed
            else if (min_ed < 350.f)  reward += 0.08f; // sweet spot
            else                      reward -= 0.04f; // too far, camping
        }

        // Penalize hugging zone edges (corner camping)
        int cx = constants::WIDTH/2, cy = constants::HEIGHT/2;
        float edge_dist = std::min({
            std::abs(p_pos.x - (cx - 300.f)),
            std::abs(p_pos.x - (cx + 300.f)),
            std::abs(p_pos.y - (cy - 300.f)),
            std::abs(p_pos.y - (cy + 300.f))
        });
        if (edge_dist < 40.f) reward -= 0.08f;

        epsilon = std::max(EPS_MIN, epsilon * EPS_DECAY);

        if (shoot_idx == 1 && player->can_shoot()) {
            if (found_enemy) {
                float   b_speed  = 10.f;
                Vector2 pred_pos = predict_aim(p_pos, near_ep, near_ev, b_speed);
                Vector2 fire_dir = (pred_pos - p_pos).normalize();
                Vector2 fire_target = p_pos + fire_dir * 1200.f;
                p_bullets.push_back(new LogicBullet(p_pos, fire_target, 10, b_speed));
                player->fire();
                shots_fired++;
                reward += 0.3f; // small bonus for attempting a shot at a real enemy
            }
        }

        bool died = update_entities(reward);
        if (died) reward = -50.0f; // death penalty

        total_reward += reward;
        step_count++;

        Vec next_state = get_state();
        Vec act_indices = {(float)move_idx, (float)shoot_idx};
        
        float lp = std::log(probs[move_idx] + 1e-9f) +
                   std::log(probs[constants::MOVE_ACTIONS + shoot_idx] + 1e-9f);

        rollout.push_back({state, act_indices, reward, next_state, lp, died});

        if ((int)rollout.size() >= train_every || died) {
            finish_episode();
        }

        // --- Curriculum Learning (E) ---
        // difficulty scales from 0.0 to 1.0 over 1M steps
        float difficulty = std::min(1.0f, (float)step_count / 1000000.f);
        int base_max_enemies = 1 + (int)(difficulty * 3); // 1 to 4 base
        int max_enemies    = std::min(8, base_max_enemies + score/500);
        
        int base_spawn_interval = 120 - (int)(difficulty * 60); // 120 to 60 base
        int spawn_interval = std::max(20, base_spawn_interval - score/300);

        if (++spawn_timer >= spawn_interval && (int)enemies.size() < max_enemies) {
            enemies.push_back(new LogicEnemy(player, e_bullets));
            spawn_timer = 0;
        }

        if (died) { alive = false; return false; }
        return true;
    }

    bool GameSim::update_entities(float& reward) {
        bool player_died = false;
        for (auto eit = enemies.begin(); eit != enemies.end();) {
            (*eit)->update(score);
            bool killed = false;
            Rect er = (*eit)->get_rect();
            for (auto bit = p_bullets.begin(); bit != p_bullets.end();) {
                Rect br = (*bit)->get_rect();
                if (rects_intersect(er, br)) {
                    (*eit)->take_damage(50);
                    reward    += 10.0f; // +10 hit
                    shots_hit += 1;
                    delete *bit; bit = p_bullets.erase(bit);
                    if (!(*eit)->is_alive() && !killed) {
                        score  += 100;
                        reward += 50.0f; // +50 kill
                        killed  = true;
                    }
                } else ++bit;
            }
            if (killed) { delete *eit; eit = enemies.erase(eit); continue; }
            Rect pr = player->get_rect();
            if (rects_intersect(er, pr)) {
                std::cout << "[Game] Over \u2014 collided with enemy\n";
                player_died = true;
            }
            ++eit;
        }
        for (auto bit = p_bullets.begin(); bit != p_bullets.end();) {
            (*bit)->update();
            if (!(*bit)->is_alive()) {
                delete *bit; bit = p_bullets.erase(bit);
            } else ++bit;
        }
        for (auto bit = e_bullets.begin(); bit != e_bullets.end();) {
            (*bit)->update();
            Rect br = (*bit)->get_rect(), pr = player->get_rect();
            if (rects_intersect(br, pr)) {
                player->take_damage(40);
                delete *bit; bit = e_bullets.erase(bit);
                if (player->get_health() <= 0) {
                    std::cout << "[Game] Over \u2014 HP zero\n";
                    player_died = true;
                }
            } else if (!(*bit)->is_alive()) {
                delete *bit; bit = e_bullets.erase(bit);
            } else ++bit;
        }
        return player_died;
    }
}

/*
OLD REWARD LOGIC AND ROLLOUT (FOR REFERENCE):
(Previously contained complex shaping rewards and conditional rollout training)

    bool GameSim::step() {
        ... (complex rewards removed) ...
        if (died) reward = -20.0f;
        ...
        if ((int)rollout.size() >= train_every) {
             // (Training logic using G_boot)
        }
    }
*/
