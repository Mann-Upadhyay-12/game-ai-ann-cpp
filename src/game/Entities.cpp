#include "Entities.hpp"
#include "../common/RNG.hpp"
#include "../common/Constants.hpp"

namespace game {
    LogicEnemy::LogicEnemy(LogicPlayer* p, std::vector<LogicBullet*>& eb)
        : LogicEntity({0,0}, 26, 26),
          player_ptr(p), bullets_ref(eb), min_dist(160.f)
    {
        int side = RNG::rng_int(0,3);
        if      (side==0) pos = {RNG::flt(0,constants::WIDTH), -35};
        else if (side==1) pos = {(float)constants::WIDTH+35, RNG::flt(0,constants::HEIGHT)};
        else if (side==2) pos = {RNG::flt(0,constants::WIDTH), (float)constants::HEIGHT+35};
        else              pos = {-35, RNG::flt(0,constants::HEIGHT)};

        spiral_speed = RNG::flt(1.5f, 4.5f) * (RNG::rng_int(0,1) ? 1 : -1) / 100.f;
        shoot_cd     = RNG::rng_int(120, 240);
        prev_pos     = pos;
        vel_smooth   = {0,0};
    }

    void LogicEnemy::update(int score) {
        Vector2 to_p = player_ptr->get_pos() - pos;
        float   dist = to_p.length();
        Vector2 dir  = to_p.normalize();

        if (dist > min_dist)            pos += dir * 2.2f;
        else if (dist < min_dist-20.f)  pos -= dir * 1.1f;

        Vector2 perp = {-dir.y, dir.x};
        pos += perp * (spiral_speed * 140.f);

        Vector2 raw = pos - prev_pos;
        vel_smooth  = vel_smooth * 0.7f + raw * 0.3f;
        prev_pos    = pos;

        if (--shoot_cd <= 0) {
            float   bspd = 5.2f;
            Vector2 p_pos = player_ptr->get_pos();
            Vector2 p_vel = player_ptr->get_vel();
            
            // Iterative prediction for better accuracy
            Vector2 pred = p_pos;
            for (int i = 0; i < 5; i++) {
                float t = pos.distance_to(pred) / bspd;
                pred = p_pos + p_vel * t;
            }

            // Score-based difficulty scaling using the user's formula pattern
            // bad_weight starts at 1.0 (score 0) and decays to 0.0 (score 3000+)
            float bad_weight = std::max(0.0f, 1.0f - (float)score / 3000.f);
            
            // target = (bad_weight * current) - (bad_weight - 1) * prediction
            // which simplifies to: bad_weight * current + (1 - bad_weight) * prediction
            Vector2 current = p_pos;
            Vector2 target = (current * bad_weight) - (pred * (bad_weight - 1.0f));

            bullets_ref.push_back(new LogicBullet(pos, target, 8, bspd));
            shoot_cd = RNG::rng_int(80, 160);
        }
    }
}
