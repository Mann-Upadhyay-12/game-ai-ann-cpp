# Shooter RL Improvements TODO

- [x] **A. Pure A2C (No Replay Buffer)**
    - Refactor `GameSim::step` and `GameSim::finish_episode` to train on each step or small rollout without a large replay buffer.
    - Keep old logic in comments at end of file.
- [x] **B. Simplified Rewards**
    - +1 survive (per step)
    - +10 hit
    - +50 kill
    - -20 death
- [x] **C. Frame Stacking**
    - Modify `get_state` to return the last 4 states concatenated.
- [x] **D. Discretize Actions**
    - 8 movement directions + shoot yes/no.
- [x] **E. Curriculum Learning**
    - Gradually increase difficulty (spawn rate, enemy speed, etc.).
- [x] **F. PPO Implementation**
    - Upgrade `PolicyNet` to support PPO (Importance sampling, Clipping).


- [x] **G. Advanced AI Architecture**
    - [x] Relative State Representation (player-centric sensing).
    - [x] Aim "Nudge" Output Head (+/- 15 degrees) for fine accuracy.
    - [x] Generalized Advantage Estimation (GAE) for cleaner training.
    - [x] Bullet Cost reward to penalize mindless spam.
    - [x] Threat-gated engagement rewards.

- [x] **H. Future Polishing**
    - [x] Action Repeating (Smooth Movement).
    - [x] Separate Advantage Heads (Survival vs Offense).
    - [x] Dynamic Zone sizing (score-dependent).

- [ ] **I. Improve Surivival**

