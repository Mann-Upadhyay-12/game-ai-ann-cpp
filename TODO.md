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


- [ ] Review the reward system
    - Add higher bullet cost
    - Try to fix rewards to make the AI dodge and also move around to look "alive" instead of just staying in corner and shooting like a turrent 
