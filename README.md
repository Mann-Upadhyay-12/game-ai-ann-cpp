# Spiral Shooter RL

A high-performance C++ implementation of an Actor-Critic (A2C) reinforcement learning agent that learns to play a top-down "Spiral Shooter" game. 

**This project is implemented from scratch in raw C++ without any external machine learning libraries like PyTorch or TensorFlow.**

## Features

- **Custom Machine Learning Engine**: Manual implementation of:
  - Neural Network layers (MLP with LeakyReLU and LayerNorm).
  - Manual Backpropagation.
  - Adam Optimizer.
- **Reinforcement Learning**:
  - Actor-Critic (A2C) architecture.
  - Advantage normalization for stable training.
  - Entropy bonus to encourage exploration.
  - Bootstrapped TD-returns.
- **Game Engine**:
  - Efficient SDL2-based visualization.
  - Headless mode for maximum training throughput (thousands of steps per second).
  - Predictive aiming logic for enemies and AI-assisted curriculum learning.
- **Persistence**: Save and load model weights and optimizer states.

## Architecture

The project is organized into a modular structure:

- `src/common`: Core utilities like `Vector2` math, Random Number Generation (RNG), and global constants.
- `src/nn`: Neural Network components, including the `PolicyNet` (Actor), `ValueNet` (Critic), and `Adam` optimizer.
- `src/game`: Game logic, entity management, and the `GameSim` RL environment.
- `src/rendering`: SDL2 rendering engine and particle systems.

## Getting Started

### Prerequisites

- C++17 compiler (e.g., GCC, Clang)
- SDL2, SDL2_image, SDL2_ttf

On Ubuntu/Debian:
```bash
sudo apt-get install libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev
```

### Building

To build the visual game:
```bash
make
```

To build the headless trainer:
```bash
make headless
```

### Running

- **AI Mode (GUI)**: `./shooter`
- **Human Mode (GUI)**: `./shooter human`
- **Headless Training**: `./shooter_headless`

## RL Details

### State Space (23 Dimensions)
- Player position and velocity.
- Nearest enemy relative position, distance, and velocity.
- Two nearest bullets: relative position, distance, velocity, and threat level (dot product).
- Weapon cooldown and player health.

### Action Space (Continuous)
1. `MOVE_X`: Horizontal movement (tanh).
2. `MOVE_Y`: Vertical movement (tanh).
3. `AIM_ANGLE`: Aiming direction (linear/clamped).
4. `SHOOT_PROB`: Probability of firing (sigmoid).

### Reward Shaping
The agent receives rewards for:
- Surviving over time.
- Staying near the center of the arena (anti-camping).
- Tracking enemies with the aim indicator.
- Successfully hitting enemies.
- Killing enemies.
Penalties are applied for:
- Taking damage or colliding with enemies.
- Being targeted by bullets.
- Wasted shots.

## Why C++?

Implementing RL from scratch in C++ provides a deep understanding of the underlying math and systems. By avoiding "black box" libraries, this project demonstrates:
1. **Manual Gradient Descent**: Every derivative is hand-calculated and implemented.
2. **Memory Management**: High-performance execution without the overhead of interpreted languages.
3. **Systems Integration**: Seamless coupling between the game simulation and the learning agent.

---

*Unequivocally "GitHub Worthy" — demonstrating systems engineering and ML fundamentals.*
