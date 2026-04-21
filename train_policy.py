#!/usr/bin/env python3
"""
train_policy.py
---------------
Deep Q-Learning for 3-drone disc coverage on a 100x100m grid.
Adapted from deep_q.py to support:
  - 3 drones (instead of 2)
  - 100x100m area (instead of 4x4m)
  - 25 disc positions from disc_positions.json
  - Grid-based action space (discretized area)
  - Saves trained policy to policy.pkl for use in policy_executor.py

Run this on your HOST machine (no ROS needed):
  python3 train_policy.py

After training, copy policy.pkl into your ROS package scripts/ folder.
"""

import numpy as np
import random
import json
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# Area bounds — must match disc_world.world and policy_executor.py
GRID_MIN = -45.0
GRID_MAX =  45.0
COVERAGE_RADIUS = 5.0       # meters — must match policy_executor.py

# Grid resolution — how many cells per axis
# 30x30 = 900 cells, each cell = 3m wide (finer than coverage radius of 5m)
GRID_SIZE = 30

# Number of drones
NUM_DRONES = 3

# Action space: each drone can pick any of GRID_SIZE^2 cells
# Total joint actions = GRID_SIZE^2 per drone (each drone acts independently)
ACTIONS_PER_DRONE = GRID_SIZE * GRID_SIZE

# Disc positions file — update path if needed
DISC_POSITIONS_PATH = "disc_positions.json"

# Training hyperparameters
LEARNING_RATE    = 0.0005
GAMMA            = 0.95
EPSILON_START    = 1.0
EPSILON_MIN      = 0.01
EPSILON_DECAY    = 0.998    # Slower decay = more exploration
NUM_EPISODES     = 3000     # More episodes
MAX_STEPS        = 100      # More steps per episode
BATCH_SIZE       = 128
MEMORY_SIZE      = 50000
TARGET_UPDATE    = 200      # More frequent target updates

# Output files
POLICY_SAVE_PATH    = "policy.pkl"
CONVERGENCE_PLOT    = "convergence.png"
COVERAGE_PLOT       = "coverage.png"


# ─────────────────────────────────────────────
# GRID UTILITIES
# ─────────────────────────────────────────────

def cell_to_xy(cell_idx):
    """Convert grid cell index to (x, y) world coordinates (cell center)."""
    row = cell_idx // GRID_SIZE
    col = cell_idx % GRID_SIZE
    cell_size = (GRID_MAX - GRID_MIN) / GRID_SIZE
    x = GRID_MIN + (col + 0.5) * cell_size
    y = GRID_MIN + (row + 0.5) * cell_size
    return x, y

def xy_to_cell(x, y):
    """Convert (x, y) world coordinates to grid cell index."""
    cell_size = (GRID_MAX - GRID_MIN) / GRID_SIZE
    col = int((x - GRID_MIN) / cell_size)
    row = int((y - GRID_MIN) / cell_size)
    col = max(0, min(GRID_SIZE - 1, col))
    row = max(0, min(GRID_SIZE - 1, row))
    return row * GRID_SIZE + col


# ─────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────

class DroneEnvironment:
    """
    3-drone disc coverage environment.
    
    State:  [cell_uav1, cell_uav2, cell_uav3] — normalized to [0,1]
    Action: [cell_uav1, cell_uav2, cell_uav3] — each drone picks a grid cell
    Reward: number of discs covered by any drone
    """

    def __init__(self, disc_positions):
        self.disc_positions = disc_positions  # list of (x, y) tuples
        self.num_discs = len(disc_positions)
        self.drone_cells = [0, 0, 0]  # current cell indices

    def reset(self):
        """Reset all drones to center of grid."""
        center_cell = xy_to_cell(0.0, 0.0)
        self.drone_cells = [center_cell] * NUM_DRONES
        return self._get_state()

    def _get_state(self):
        """Return normalized state vector."""
        return np.array(self.drone_cells, dtype=np.float32) / ACTIONS_PER_DRONE

    def _compute_coverage(self, drone_xys):
        """Count how many discs are covered by at least one drone."""
        covered = 0
        for dx, dy in self.disc_positions:
            for ux, uy in drone_xys:
                dist = np.sqrt((dx - ux)**2 + (dy - uy)**2)
                if dist <= COVERAGE_RADIUS:
                    covered += 1
                    break
        return covered

    def step(self, actions):
        """
        Apply actions (list of 3 cell indices) and return (next_state, reward).
        This is a placement problem — each step places all 3 drones.
        """
        self.drone_cells = list(actions)
        drone_xys = [cell_to_xy(c) for c in actions]
        covered = self._compute_coverage(drone_xys)
        reward = covered  # reward = number of discs covered
        return self._get_state(), reward

    def get_drone_positions(self):
        """Return current drone (x, y) positions."""
        return [cell_to_xy(c) for c in self.drone_cells]


# ─────────────────────────────────────────────
# NEURAL NETWORK
# ─────────────────────────────────────────────

class DQN(nn.Module):
    """
    Deep Q-Network.
    Input:  state (3 normalized cell indices)
    Output: Q-values for each possible action per drone
    
    Each drone has its own output head — decentralized execution.
    """

    def __init__(self, state_dim=3, actions_per_drone=ACTIONS_PER_DRONE):
        super(DQN, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        # Separate head for each drone
        self.head1 = nn.Linear(256, actions_per_drone)
        self.head2 = nn.Linear(256, actions_per_drone)
        self.head3 = nn.Linear(256, actions_per_drone)

    def forward(self, x):
        shared = self.shared(x)
        q1 = self.head1(shared)
        q2 = self.head2(shared)
        q3 = self.head3(shared)
        return q1, q2, q3


# ─────────────────────────────────────────────
# REPLAY BUFFER
# ─────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, state, actions, reward, next_state):
        self.buffer.append((state, actions, reward, next_state))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)).to(self.device),
            torch.LongTensor(np.array(actions)).to(self.device),
            torch.FloatTensor(rewards).to(self.device),
            torch.FloatTensor(np.array(next_states)).to(self.device),
        )

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────
# AGENT
# ─────────────────────────────────────────────

class Agent:
    def __init__(self, disc_positions):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.env = DroneEnvironment(disc_positions)
        self.disc_positions = disc_positions

        self.epsilon = EPSILON_START
        self.q_network = DQN().to(self.device)
        self.target_network = DQN().to(self.device)
        self._update_target()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.MSELoss()
        self.memory = ReplayBuffer(MEMORY_SIZE, self.device)

        total_params = sum(p.numel() for p in self.q_network.parameters())
        print(f"Network parameters: {total_params:,}")

    def _update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def get_actions(self, state):
        """Epsilon-greedy action selection for all 3 drones."""
        if random.uniform(0, 1) < self.epsilon:
            return [random.randint(0, ACTIONS_PER_DRONE - 1) for _ in range(NUM_DRONES)]
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q1, q2, q3 = self.q_network(state_tensor)
        return [q1.argmax().item(), q2.argmax().item(), q3.argmax().item()]

    def update(self):
        if len(self.memory) < BATCH_SIZE:
            return None

        states, actions, rewards, next_states = self.memory.sample(BATCH_SIZE)

        # Current Q values for each drone
        q1, q2, q3 = self.q_network(states)
        curr_q1 = q1.gather(1, actions[:, 0].unsqueeze(1)).squeeze(1)
        curr_q2 = q2.gather(1, actions[:, 1].unsqueeze(1)).squeeze(1)
        curr_q3 = q3.gather(1, actions[:, 2].unsqueeze(1)).squeeze(1)

        # Target Q values
        with torch.no_grad():
            nq1, nq2, nq3 = self.target_network(next_states)
            target_q = rewards + GAMMA * (
                nq1.max(1)[0] + nq2.max(1)[0] + nq3.max(1)[0]
            ) / NUM_DRONES

        loss = (
            self.loss_fn(curr_q1, target_q) +
            self.loss_fn(curr_q2, target_q) +
            self.loss_fn(curr_q3, target_q)
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self):
        print("\n" + "=" * 60)
        print("Starting Deep Q-Learning Training")
        print(f"Grid: {GRID_SIZE}x{GRID_SIZE} cells over {GRID_MAX-GRID_MIN:.0f}x{GRID_MAX-GRID_MIN:.0f}m area")
        print(f"Discs: {len(self.disc_positions)} | Coverage radius: {COVERAGE_RADIUS}m")
        print(f"Episodes: {NUM_EPISODES} | Steps/episode: {MAX_STEPS}")
        print("=" * 60 + "\n")

        episode_rewards = []
        losses = []
        best_reward = 0
        best_actions = None
        total_steps = 0

        for episode in range(NUM_EPISODES):
            state = self.env.reset()
            total_reward = 0
            total_loss = 0
            best_episode_reward = 0
            best_episode_actions = None

            for step in range(MAX_STEPS):
                actions = self.get_actions(state)
                next_state, reward = self.env.step(actions)
                self.memory.push(state, actions, reward, next_state)

                total_reward += reward
                state = next_state
                total_steps += 1

                if reward > best_episode_reward:
                    best_episode_reward = reward
                    best_episode_actions = actions[:]

                step_loss = self.update()
                if step_loss is not None:
                    total_loss += step_loss

                if total_steps % TARGET_UPDATE == 0:
                    self._update_target()

            # Track best overall result
            if best_episode_reward > best_reward:
                best_reward = best_episode_reward
                best_actions = best_episode_actions
                positions = [cell_to_xy(a) for a in best_actions]
                print(f"  ★ New best: {best_reward}/{len(self.disc_positions)} discs covered")
                print(f"    UAV1: {positions[0]} | UAV2: {positions[1]} | UAV3: {positions[2]}")

            self.epsilon = max(self.epsilon * EPSILON_DECAY, EPSILON_MIN)
            episode_rewards.append(total_reward)
            losses.append(total_loss)

            if episode % 50 == 0:
                print(f"Episode {episode:4d} | Reward: {total_reward:6.1f} | "
                      f"Best: {best_reward}/{len(self.disc_positions)} | "
                      f"Loss: {total_loss:.4f} | Epsilon: {self.epsilon:.3f}")

        print(f"\nTraining complete. Best coverage: {best_reward}/{len(self.disc_positions)} discs")
        return episode_rewards, losses, best_actions

    def save_policy(self, best_actions):
        """
        Save the trained policy to a pickle file.
        The policy is stored as:
          - model weights (for inference)
          - best_actions (optimal cell indices)
          - best_positions (optimal x,y positions)
          - disc_positions (for reference)
          - grid config (for cell_to_xy reconstruction)
        """
        best_positions = [cell_to_xy(a) for a in best_actions]
        covered = self.env._compute_coverage(best_positions)

        policy_data = {
            "model_state_dict": self.q_network.state_dict(),
            "best_actions": best_actions,
            "best_positions": best_positions,
            "coverage": covered,
            "disc_positions": self.disc_positions,
            "grid_config": {
                "grid_min": GRID_MIN,
                "grid_max": GRID_MAX,
                "grid_size": GRID_SIZE,
                "coverage_radius": COVERAGE_RADIUS,
            }
        }

        with open(POLICY_SAVE_PATH, "wb") as f:
            pickle.dump(policy_data, f)

        print(f"\nPolicy saved to {POLICY_SAVE_PATH}")
        print(f"Best positions:")
        for i, (x, y) in enumerate(best_positions):
            print(f"  UAV{i+1}: ({x:.2f}, {y:.2f})")
        print(f"Coverage: {covered}/{len(self.disc_positions)} discs "
              f"({100*covered/len(self.disc_positions):.1f}%)")

    def plot_convergence(self, episode_rewards, losses):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(episode_rewards)
        axes[0].set_title('Episode Rewards')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Total Reward')
        axes[0].grid(True)
        axes[1].plot(losses, color='orange')
        axes[1].set_title('Training Loss')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Total Loss')
        axes[1].grid(True)
        plt.tight_layout()
        plt.savefig(CONVERGENCE_PLOT, dpi=150)
        print(f"Saved convergence plot to {CONVERGENCE_PLOT}")

    def plot_coverage(self, best_actions):
        positions = [cell_to_xy(a) for a in best_actions]
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot all discs
        for dx, dy in self.disc_positions:
            ax.scatter(dx, dy, color='gray', s=40, zorder=2)

        colors = ['red', 'blue', 'green']
        covered = set()

        for i, (ux, uy) in enumerate(positions):
            circle = plt.Circle((ux, uy), COVERAGE_RADIUS,
                                 alpha=0.25, color=colors[i],
                                 label=f'UAV{i+1} ({ux:.1f}, {uy:.1f})')
            ax.add_patch(circle)
            ax.scatter(ux, uy, color=colors[i], marker='X', s=200, zorder=4)

            for j, (dx, dy) in enumerate(self.disc_positions):
                if np.sqrt((dx-ux)**2 + (dy-uy)**2) <= COVERAGE_RADIUS:
                    covered.add(j)

        # Highlight covered discs
        for j in covered:
            dx, dy = self.disc_positions[j]
            ax.scatter(dx, dy, color='lime', s=80, zorder=3)

        ax.set_xlim(GRID_MIN - 5, GRID_MAX + 5)
        ax.set_ylim(GRID_MIN - 5, GRID_MAX + 5)
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_title(f'Optimal Drone Coverage — {len(covered)}/{len(self.disc_positions)} discs covered')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(COVERAGE_PLOT, dpi=150)
        print(f"Saved coverage plot to {COVERAGE_PLOT}")


# ─────────────────────────────────────────────
# POLICY LOADER (used by policy_executor.py)
# ─────────────────────────────────────────────

def load_policy(policy_path=POLICY_SAVE_PATH):
    """
    Load a saved policy and return a callable.
    
    Usage in policy_executor.py:
        from train_policy import load_policy
        policy_fn = load_policy("policy.pkl")
        actions = policy_fn(state, disc_positions)
    """
    with open(policy_path, "rb") as f:
        data = pickle.load(f)

    best_positions = data["best_positions"]
    uav_names = ["uav1", "uav2", "uav3"]

    def trained_policy(state, disc_positions):
        """Return the optimal fixed positions learned during training."""
        return {uav: pos for uav, pos in zip(uav_names, best_positions)}

    print(f"Loaded policy — coverage: {data['coverage']}/{len(data['disc_positions'])} discs")
    return trained_policy


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Load disc positions
    if os.path.exists(DISC_POSITIONS_PATH):
        with open(DISC_POSITIONS_PATH, "r") as f:
            data = json.load(f)
        disc_positions = [(p[0], p[1]) for p in data["positions"]]
        print(f"Loaded {len(disc_positions)} disc positions from {DISC_POSITIONS_PATH}")
    else:
        print(f"ERROR: {DISC_POSITIONS_PATH} not found.")
        print("Copy disc_positions.json from your DiscWorld repo to this directory.")
        exit(1)

    # Train
    agent = Agent(disc_positions)
    episode_rewards, losses, best_actions = agent.train()

    # Save policy
    agent.save_policy(best_actions)

    # Plot results
    print("\nGenerating plots...")
    agent.plot_convergence(episode_rewards, losses)
    agent.plot_coverage(best_actions)

    print("\nDone! Next steps:")
    print("  1. Copy policy.pkl to your ROS package scripts/ folder")
    print("  2. Update policy_executor.py to use load_policy() instead of random policy")
    print("     from train_policy import load_policy")
    print("     policy_fn = load_policy('policy.pkl')")
