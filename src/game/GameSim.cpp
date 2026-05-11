#include "GameSim.hpp"
#include "../common/RNG.hpp"
#include "../common/Constants.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cassert>
#include <cmath>

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

        std::vector<Vec> states, actions, next_states;
        std::vector<float> rewards_s, rewards_o, log_probs;
        std::vector<bool> dones;
        states.reserve(rollout.size());
        actions.reserve(rollout.size());
        next_states.reserve(rollout.size());
        rewards_s.reserve(rollout.size());
        rewards_o.reserve(rollout.size());
        log_probs.reserve(rollout.size());
        dones.reserve(rollout.size());

        for (auto& step : rollout) {
            states.push_back(step.state);
            actions.push_back(step.action);
            next_states.push_back(step.next_state);
            rewards_s.push_back(step.reward_survival);
            rewards_o.push_back(step.reward_offense);
            log_probs.push_back(step.log_prob);
            dones.push_back(step.done);
        }
        
        net->train_ppo(states, actions, rewards_s, rewards_o, next_states, dones, log_probs);
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
        Vector2 p_pos = player->get_pos();
        // 1. Player relative position (to center) and absolute velocity
        s.push_back((p_pos.x - constants::WIDTH/2) / (constants::WIDTH/2));
        s.push_back((p_pos.y - constants::HEIGHT/2) / (constants::HEIGHT/2));
        s.push_back(player->get_vel().x / 8.f);
        s.push_back(player->get_vel().y / 8.f);

        // 2. Nearest Enemy (Relative)
        float   min_ed = 1e9f;
        Vector2 near_ep = {0,0}, near_ev = {0,0};
        bool found_enemy = false;
        for (auto* e : enemies) {
            if (!is_visible(e->get_pos())) continue;
            float d = p_pos.distance_to(e->get_pos());
            if (d < min_ed) { min_ed = d; near_ep = e->get_pos(); near_ev = e->get_vel_smooth(); found_enemy = true; }
        }
        if (found_enemy) {
            Vector2 e_rel = (near_ep - p_pos) / 500.f; // scaled relative pos
            s.push_back(e_rel.x);
            s.push_back(e_rel.y);
            s.push_back(std::min(1.f, min_ed / 900.f));
            s.push_back(near_ev.x / 5.f);
            s.push_back(near_ev.y / 5.f);
        } else {
            s.push_back(0.f); s.push_back(0.f); s.push_back(1.f); s.push_back(0.f); s.push_back(0.f);
        }

        // 3. Most Threatening Bullets (Relative)
        struct BulletInfo { float dist; Vector2 rel_pos, vel; float threat; };
        std::vector<BulletInfo> found_bullets;
        for (auto* b : e_bullets) {
            if (!is_visible(b->get_pos())) continue;
            float d = p_pos.distance_to(b->get_pos());
            Vector2 rel_p = b->get_pos() - p_pos;
            Vector2 to_p = (-rel_p).normalize();
            Vector2 bvn  = b->get_vel().normalize();
            float threat = bvn.dot(to_p);
            found_bullets.push_back({d, rel_p, b->get_vel(), threat});
        }
        std::sort(found_bullets.begin(), found_bullets.end(), [](const BulletInfo& a, const BulletInfo& b) {
            if (std::abs(a.threat - b.threat) > 0.1f) return a.threat > b.threat;
            return a.dist < b.dist;
        });

        for (int i = 0; i < 4; i++) {
            if (i < (int)found_bullets.size()) {
                Vector2 rel_n = found_bullets[i].rel_pos / 500.f;
                s.push_back(rel_n.x);
                s.push_back(rel_n.y);
                
                float b_speed = found_bullets[i].vel.length();
                float tti = (b_speed > 0.01f) ? found_bullets[i].dist / b_speed : 999.f;
                s.push_back(std::min(1.f, tti / 60.f));

                Vector2 bvn = found_bullets[i].vel.normalize();
                s.push_back(bvn.x);
                s.push_back(bvn.y);
                s.push_back(found_bullets[i].threat);
            } else {
                s.push_back(0.f); s.push_back(0.f); s.push_back(1.f);
                s.push_back(0.f); s.push_back(0.f); s.push_back(-1.f);
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
        int move_idx, nudge_idx, shoot_idx;
        float log_prob;

        if (repeat_timer <= 0) {
            Vec probs = net->forward(state);
            move_idx  = RNG::categorical(probs, 0, constants::MOVE_ACTIONS);
            nudge_idx = RNG::categorical(probs, constants::MOVE_ACTIONS, constants::NUDGE_ACTIONS);
            shoot_idx = RNG::categorical(probs, constants::MOVE_ACTIONS + constants::NUDGE_ACTIONS, constants::SHOOT_ACTIONS);

            log_prob = std::log(probs[move_idx] + 1e-9f) +
                       std::log(probs[constants::MOVE_ACTIONS + nudge_idx] + 1e-9f) +
                       std::log(probs[constants::MOVE_ACTIONS + constants::NUDGE_ACTIONS + shoot_idx] + 1e-9f);

            last_move_idx  = move_idx;
            last_nudge_idx = nudge_idx;
            last_shoot_idx = shoot_idx;
            last_log_prob  = log_prob;
            repeat_timer   = action_repeat;
        } else {
            move_idx  = last_move_idx;
            nudge_idx = last_nudge_idx;
            shoot_idx = last_shoot_idx;
            log_prob  = last_log_prob;
            repeat_timer--;
        }

        // Interpretation
        static const Vector2 dirs[] = {
            {0,-1}, {0.707f,-0.707f}, {1,0}, {0.707f,0.707f},
            {0,1}, {-0.707f,0.707f}, {-1,0}, {-0.707f,-0.707f}, {0,0}
        };
        Vector2 move_dir = dirs[move_idx];
        player->move(move_dir, true);

        float reward_s = 0.1f, reward_o = 0.f; // Base survival reward per step
        Vector2 p_pos = player->get_pos();

        // --- Movement rewards ---
        if (move_idx == 8) reward_s -= 0.05f;

        // --- Dodging Reward ---
        float max_threat = 0.f;
        Vector2 best_bvn = {0,0};
        float dist_at_max_threat = 1e9f;
        for (auto* b : e_bullets) {
            if (!is_visible(b->get_pos())) continue;
            float d = p_pos.distance_to(b->get_pos());
            Vector2 to_p = (p_pos - b->get_pos()).normalize();
            Vector2 bvn  = b->get_vel().normalize();
            float threat = bvn.dot(to_p);
            if (threat > max_threat) {
                max_threat = threat;
                best_bvn = bvn;
                dist_at_max_threat = d;
            }
        }

        float   min_ed = 1e9f;
        Vector2 near_ep = {0,0}, near_ev = {0,0};
        bool    found_enemy = false;
        for (auto* e : enemies) {
            if (!is_visible(e->get_pos())) continue;
            float d = p_pos.distance_to(e->get_pos());
            if (d < min_ed) { min_ed = d; near_ep = e->get_pos(); near_ev = e->get_vel_smooth(); found_enemy = true; }
        }

        if (found_enemy) {
            if (max_threat < 0.5f) {
                if (min_ed < 120.f)       reward_s -= 0.1f;
                else if (min_ed < 350.f)  reward_s += 0.04f;
                else                      reward_s -= 0.02f;
            }
        }

        if (max_threat > 0.7f && move_idx != 8) {
            float dodge_quality = 1.0f - std::abs(move_dir.dot(best_bvn));
            float urgency = 1.f - std::min(1.f, dist_at_max_threat / 250.f);
            reward_s += 0.15f * dodge_quality * urgency * max_threat;
        }

        // --- Dynamic Zone sizing ---
        zone_size = std::max(150.f, 350.f - score / 20.f);
        int cx = constants::WIDTH/2, cy = constants::HEIGHT/2;
        float dx = std::abs(p_pos.x - cx), dy = std::abs(p_pos.y - cy);
        if (dx > zone_size || dy > zone_size) {
            reward_s -= 0.1f; // Out of zone penalty
        }

        epsilon = std::max(EPS_MIN, epsilon * EPS_DECAY);

        if (shoot_idx == 1 && player->can_shoot()) {
            if (found_enemy) {
                float   b_speed  = 10.f;
                Vector2 pred_pos = predict_aim(p_pos, near_ep, near_ev, b_speed);
                Vector2 fire_dir = (pred_pos - p_pos).normalize();

                float nudge_deg = (nudge_idx - 3) * 5.0f; 
                float nudge_rad = nudge_deg * (3.14159f / 180.f);
                float cs = std::cos(nudge_rad), sn = std::sin(nudge_rad);
                Vector2 nudged_dir = { fire_dir.x * cs - fire_dir.y * sn,
                                       fire_dir.x * sn + fire_dir.y * cs };

                Vector2 fire_target = p_pos + nudged_dir * 1200.f;
                p_bullets.push_back(new LogicBullet(p_pos, fire_target, 10, b_speed));
                player->fire();
                shots_fired++;
                
                // Reward Shaping (Offense)
                reward_o -= 0.1f; // Bullet cost
                // Trust-predictor bonus: reward for small nudges
                reward_o += 0.05f * (1.0f - std::abs(nudge_deg) / 15.0f);
                // Engagement bonus: reward for just shooting at an enemy to encourage discovery
                reward_o += 0.02f; 
            }
        }

        bool died = update_entities(reward_s, reward_o);
        if (died) reward_s -= 50.0f;

        total_reward += (reward_s + reward_o);
        step_count++;

        Vec next_state = get_state();
        Vec act_indices = {(float)move_idx, (float)nudge_idx, (float)shoot_idx};
        
        rollout.push_back({state, act_indices, reward_s, reward_o, next_state, log_prob, died});

        if ((int)rollout.size() >= train_every || died) {
            finish_episode();
        }

        // --- Curriculum Learning ---
        float difficulty = std::min(1.0f, (float)step_count / 1000000.f);
        int base_max_enemies = 1 + (int)(difficulty * 3);
        int max_enemies    = std::min(8, base_max_enemies + score/500);
        
        int base_spawn_interval = 120 - (int)(difficulty * 60);
        int spawn_interval = std::max(20, base_spawn_interval - score/300);

        if (++spawn_timer >= spawn_interval && (int)enemies.size() < max_enemies) {
            enemies.push_back(new LogicEnemy(player, e_bullets));
            spawn_timer = 0;
        }

        if (died) { alive = false; return false; }
        return true;
    }

    bool GameSim::update_entities(float& r_s, float& r_o) {
        bool player_died = false;
        for (auto eit = enemies.begin(); eit != enemies.end();) {
            (*eit)->update(score);
            bool killed = false;
            Rect er = (*eit)->get_rect();
            for (auto bit = p_bullets.begin(); bit != p_bullets.end();) {
                Rect br = (*bit)->get_rect();
                if (rects_intersect(er, br)) {
                    (*eit)->take_damage(50);
                    r_o       += 10.0f; // +10 hit (Offense)
                    shots_hit += 1;
                    delete *bit; bit = p_bullets.erase(bit);
                    if (!(*eit)->is_alive() && !killed) {
                        score  += 100;
                        r_o    += 50.0f; // +50 kill (Offense)
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
