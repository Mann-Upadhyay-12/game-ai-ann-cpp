#pragma once
#include <cmath>

struct Vector2 {
    float x = 0, y = 0;
    Vector2(float x = 0, float y = 0) : x(x), y(y) {}
    Vector2 operator+(const Vector2& v) const { return {x+v.x, y+v.y}; }
    Vector2 operator-(const Vector2& v) const { return {x-v.x, y-v.y}; }
    Vector2 operator*(float s)          const { return {x*s,   y*s};   }
    Vector2 operator/(float s)          const { return {x/s,   y/s};   }
    Vector2& operator+=(const Vector2& v){ x+=v.x; y+=v.y; return *this; }
    Vector2& operator-=(const Vector2& v){ x-=v.x; y-=v.y; return *this; }
    float length()   const { return std::sqrt(x*x + y*y); }
    float dot(const Vector2& v) const { return x*v.x + y*v.y; }
    Vector2 normalize() const {
        float l = length();
        return (l > 1e-6f) ? Vector2{x/l, y/l} : Vector2{0,0};
    }
    float distance_to(const Vector2& v) const { return (*this - v).length(); }
};
