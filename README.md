# Drone Simulation Q-Learning

A reinforcement learning framework for coordinated multi-drone disc coverage, built on top of the [MRS UAV System](https://github.com/ctu-mrs/mrs_uav_system) and [Fly4Future](https://fly4future.com/) simulation stack. Three F450 drones are trained using Deep Q-Learning to find optimal positions that maximize coverage of 25 randomly scattered ground targets in a 100×100m Gazebo simulation environment.

---

## Overview

This project implements a 3-phase pipeline:

**Phase 1 — Policy Execution Framework**
A ROS node (`policy_executor.py`) reads real-time drone positions via odometry, queries a policy, and sends goto commands to all 3 drones via the MRS control manager. The policy interface is modular — swapping in a trained policy requires changing only one function.

**Phase 2 — Offline Training + Policy Deployment** *(in progress)*
A Deep Q-Network (`train_policy.py`) is trained offline against the fixed disc layout in `disc_positions.json`. The trained policy is saved as `policy.pkl` and loaded into the ROS node for deployment in the realistic simulation.

**Phase 3 — Adaptive Policy with Sweep** *(planned)*
Drones first sweep the zone to build a disc map using onboard RGBD cameras and YOLO detection. The adaptive policy then guides drones to optimal positions based on the observed layout.

---

## Repository Structure

```
├── scripts/
│   └── policy_executor.py      # ROS node: reads positions, runs policy, sends goto commands
├── launch/
│   └── ...                     # ROS launch files
├── config/
│   └── ...                     # MRS config files (world, network, custom)
├── tmux/
│   └── start.sh                # Tmux session launcher
├── train_policy.py             # Offline Deep Q-Learning training script
├── disc_positions.json         # 25 disc positions (100x100m, seed=42)
├── session.yml                 # Tmuxinator session for 3-drone simulation
├── charlotte.world             # Original Gazebo world with disc models
├── f450.sdf (1).jinja          # Modified F450 drone with front RGBD camera
├── CMakeLists.txt
└── package.xml
```

---

## System Requirements

- Ubuntu 24.04
- [MRS Apptainer](https://github.com/ctu-mrs/mrs_apptainer) (ROS Noetic container)
- NVIDIA GPU (tested on RTX 3050)
- Python 3.10+ with PyTorch, NumPy, Matplotlib

---

## Simulation Setup

### 1. Enter the Apptainer container

```bash
cd ~/mrs_apptainer
./wrapper.sh
source ~/user_ros_workspace/devel/setup.bash
```

### 2. Copy the world file

```bash
mkdir -p ~/.gazebo/worlds
cp ~/user_ros_workspace/DiscWorld/disc_world.world ~/.gazebo/worlds/disc_world.world
```

### 3. Launch the three-drone simulation

```bash
cd ~/user_ros_workspace/three_drones
./start.sh
```

### 4. Run the policy executor (new container tab)

```bash
cd ~/mrs_apptainer && ./wrapper.sh
source ~/user_ros_workspace/devel/setup.bash
rosrun example_multi_uav_coordination policy_executor.py
```

---

## Training the Policy

Run training on your host machine (no ROS required):

```bash
cd ~/Drone-Simulation-Q-Learning

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib numpy

# Train
python3 train_policy.py
```

Training runs for 1000 episodes on a 20×20 grid discretization of the 100×100m area. Output files:

| File | Description |
|---|---|
| `policy.pkl` | Saved policy with optimal drone positions |
| `convergence.png` | Episode rewards and loss curves |
| `coverage.png` | Visualization of optimal drone placement |

### Deploying the Trained Policy

After training, copy `policy.pkl` to your ROS package and update `policy_executor.py`:

```bash
cp policy.pkl ~/mrs_apptainer/user_ros_workspace/src/Drone-Simulation-Q-Learning/scripts/
```

In `policy_executor.py`, replace the `policy()` function with:

```python
from train_policy import load_policy
_trained_policy = load_policy("policy.pkl")

def policy(state, disc_positions):
    return _trained_policy(state, disc_positions)
```

---

## Policy Executor

The `policy_executor.py` node implements the full Phase 1 framework:

- Subscribes to `/uav1/hw_api/odometry`, `/uav2/hw_api/odometry`, `/uav3/hw_api/odometry`
- Calls `/uavX/control_manager/goto` services to move drones
- Computes disc coverage at each step and publishes to `/policy_executor/coverage_status`
- Runs at 0.2 Hz (configurable) to give drones time to reach targets

The `policy()` function is the only thing that needs to change between a random policy and a trained one:

```python
def policy(state, disc_positions):
    # Replace this body with your trained policy
    # Input:  state — {uav_name: (x, y, z)}
    # Output: actions — {uav_name: (x, y)}
    ...
```

---

## Disc World

The simulation uses a custom Gazebo world (`disc_world.world`) with 25 red flat cylinders randomly scattered across a 100×100m area (seed=42, minimum separation 4m). Disc positions are stored in `disc_positions.json` for use by the training script and policy executor.

To regenerate with a different seed or layout, run `generate_world.py` from the [DiscWorld repo](https://github.com/Sujjay02/DiscWorld).

---

## Related Repositories

| Repo | Description |
|---|---|
| [DiscWorld](https://github.com/Sujjay02/DiscWorld) | Gazebo world file and disc position generator |
| [disk-position](https://github.com/Sujjay02/disk-position) | Original 2-drone Deep Q-Learning prototype |
| [MRS UAV System](https://github.com/ctu-mrs/mrs_uav_system) | Core UAV autonomy stack |
| [SmartNet Lab Simulation](https://github.com/SmartNetLaboratory/RealisticSimulation-Coordinated-Target-Searching) | Base simulation package |

---

## References

- [Fly4Future Simulation Docs](https://documentation.fly4future.com/docs/simulation/)
- [MRS UAV Documentation](https://ctu-mrs.github.io/docs/introduction/)
- [ROS Noetic](https://wiki.ros.org/noetic)
