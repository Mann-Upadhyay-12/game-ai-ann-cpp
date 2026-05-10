#pragma once
#include "../common/Vector2.hpp"
#include "../common/Constants.hpp"
#include <vector>

namespace game {
    struct Rect { int x, y, w, h; };
    inline bool rects_intersect(const Rect& a, const Rect& b) {
        return !(a.x+a.w <= b.x || b.x+b.w <= a.x ||
                 a.y+a.h <= b.y || b.y+b.h <= a.y);
    }

    class LogicEntity {
    protected:
        Vector2 pos;
        int     w, h;
        bool    alive = true;
    public:
        LogicEntity(Vector2 p, int w, int h) : pos(p), w(w), h(h) {}
        virtual ~LogicEntity() = default;
        virtual void update(int score = 0) = 0;
        bool    is_alive()  const { return alive; }
        void    die()             { alive = false; }
        Vector2 get_pos()   const { return pos; }
        Rect    get_rect()  const { return { (int)(pos.x-w/2), (int)(pos.y-h/2), w, h }; }
    };

    class LogicBullet : public LogicEntity {
        Vector2 vel;
    public:
        LogicBullet(Vector2 p, Vector2 target, int sz, float spd)
            : LogicEntity(p, sz, sz) {
            vel = (target - p).normalize() * spd;
        }
        void update(int score = 0) override {
            pos += vel;
            if (pos.x < -50 || pos.x > constants::WIDTH+50 || pos.y < -50 || pos.y > constants::HEIGHT+50) die();
        }
        Vector2 get_vel() const { return vel; }
    };

    class LogicPlayer : public LogicEntity {
        Vector2 velocity;
        int     health   = 100;
        int     shoot_cd = 0;
        float   speed    = 6.f;
    public:
        LogicPlayer() : LogicEntity({constants::WIDTH/2.f, constants::HEIGHT/2.f}, 28, 28) {}
        void update(int score = 0) override { if (shoot_cd > 0) shoot_cd--; }
        void move(Vector2 dir, bool restricted = false) {
            if (dir.length() > 0) {
                if (dir.length() > 1.f) dir = dir.normalize();
                velocity = dir * speed;
                pos += velocity;
            } else { velocity = {0,0}; }
            if (restricted) {
                int cx = constants::WIDTH/2, cy = constants::HEIGHT/2;
                do_clamp(cx-300, cx+300, cy-300, cy+300);
            } else do_clamp(0, constants::WIDTH, 0, constants::HEIGHT);
        }
        void do_clamp(int x0, int x1, int y0, int y1) {
            if (pos.x < x0+w/2) pos.x = (float)(x0+w/2);
            if (pos.x > x1-w/2) pos.x = (float)(x1-w/2);
            if (pos.y < y0+h/2) pos.y = (float)(y0+h/2);
            if (pos.y > y1-h/2) pos.y = (float)(y1-h/2);
        }
        bool    can_shoot()   const { return shoot_cd == 0; }
        void    fire()              { shoot_cd = 8; }
        void    take_damage(int d)  { health -= d; }
        int     get_health()  const { return health; }
        int     get_cd()      const { return shoot_cd; }
        Vector2 get_vel()     const { return velocity; }
    };

    class LogicEnemy : public LogicEntity {
        LogicPlayer*               player_ptr;
        std::vector<LogicBullet*>& bullets_ref;
        float   spiral_speed;
        float   min_dist;
        int     health   = 150;
        int     shoot_cd;
        Vector2 prev_pos;
        Vector2 vel_smooth;
    public:
        LogicEnemy(LogicPlayer* p, std::vector<LogicBullet*>& eb);
        void update(int score = 0) override;
        void    take_damage(int d)     { health -= d; if (health <= 0) die(); }
        int     get_health()     const { return health; }
        Vector2 get_vel_smooth() const { return vel_smooth; }
    };
}
