#pragma once

namespace constants {
    const int WIDTH  = 1600;
    const int HEIGHT = 900;

    enum Action { MOVE_X = 0, MOVE_Y, AIM_ANGLE, SHOOT_PROB, NUM_ACTIONS_CONT };
    
    static constexpr int MOVE_ACTIONS = 9; // 8 dirs + stay
    static constexpr int AIM_ACTIONS  = 1; // remove — collapse to 1 (dummy, unused)
    static constexpr int SHOOT_ACTIONS = 2; // yes/no
    static constexpr int NUM_ACTIONS = MOVE_ACTIONS + SHOOT_ACTIONS;

    static constexpr int SINGLE_STATE_DIM = 35;

    static constexpr int FRAME_STACK = 4;
    static constexpr int STATE_DIM = SINGLE_STATE_DIM * FRAME_STACK;

}
