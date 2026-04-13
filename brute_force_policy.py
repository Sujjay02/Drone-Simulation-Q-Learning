#!/usr/bin/env python3
"""
brute_force_policy.py
---------------------
Finds the optimal placement for 3 drones to maximize disc coverage.
Uses exhaustive search over a grid — guaranteed to find the best solution.
Runs in seconds on CPU. No GPU needed.

Usage:
    python3 brute_force_policy.py

Output:
    policy.pkl  — ready to use in policy_executor.py
    coverage.png — visualization of optimal placement
"""

import numpy as np
import json
import pickle
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import combinations

# ─────────────────────────────────────────────
# CONFIGURATION — must match policy_executor.py
# ─────────────────────────────────────────────
GRID_MIN        = -45.0
GRID_MAX        =  45.0
COVERAGE_RADIUS =  5.0
GRID_SIZE       =  30       # 30x30 = 900 candidate positions
NUM_DRONES      =  3
DISC_POSITIONS_PATH = "disc_positions.json"
POLICY_SAVE_PATH    = "policy.pkl"
COVERAGE_PLOT       = "coverage.png"


# ─────────────────────────────────────────────
# GRID SETUP
# ─────────────────────────────────────────────

def build_grid():
    """Build all candidate (x, y) positions on the grid."""
    cell_size = (GRID_MAX - GRID_MIN) / GRID_SIZE
    positions = []
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            x = GRID_MIN + (col + 0.5) * cell_size
            y = GRID_MIN + (row + 0.5) * cell_size
            positions.append((x, y))
    return positions


def compute_coverage_matrix(grid_positions, disc_positions):
    """
    Precompute which discs each grid cell covers.
    Returns a list of sets: coverage_matrix[i] = set of disc indices covered by cell i.
    This makes the exhaustive search much faster.
    """
    coverage_matrix = []
    disc_array = np.array(disc_positions)
    for (gx, gy) in grid_positions:
        dists = np.sqrt((disc_array[:, 0] - gx)**2 + (disc_array[:, 1] - gy)**2)
        covered = set(np.where(dists <= COVERAGE_RADIUS)[0])
        coverage_matrix.append(covered)
    return coverage_matrix


# ─────────────────────────────────────────────
# EXHAUSTIVE SEARCH
# ─────────────────────────────────────────────

def exhaustive_search(coverage_matrix, num_drones, num_discs):
    """
    Try all combinations of num_drones grid cells.
    Returns (best_cells, best_coverage_count).
    
    For 30x30 grid with 3 drones: C(900, 3) = ~121 million combinations.
    With numpy set union this runs in ~30-60 seconds.
    
    We use a smarter approach: only keep cells that cover at least 1 disc,
    which dramatically reduces the search space.
    """
    # Filter to only cells that cover at least one disc
    useful_cells = [i for i, cov in enumerate(coverage_matrix) if len(cov) > 0]
    print(f"Grid cells covering at least 1 disc: {len(useful_cells)} / {len(coverage_matrix)}")

    best_count = 0
    best_combo = None
    total = len(useful_cells)
    checked = 0
    start = time.time()

    # Convert sets to frozensets for faster union
    cov = [coverage_matrix[i] for i in useful_cells]

    for i in range(total):
        for j in range(i, total):
            union_ij = cov[i] | cov[j]
            if len(union_ij) + (num_discs - len(union_ij)) < best_count:
                continue  # Pruning: can't beat best even with perfect 3rd drone
            for k in range(j, total):
                count = len(union_ij | cov[k])
                if count > best_count:
                    best_count = count
                    best_combo = (useful_cells[i], useful_cells[j], useful_cells[k])
                    elapsed = time.time() - start
                    print(f"  ★ New best: {best_count}/{num_discs} discs "
                          f"(cells {best_combo}, t={elapsed:.1f}s)")
                    if best_count == num_discs:
                        return best_combo, best_count

        checked += total - i
        if i % 20 == 0:
            elapsed = time.time() - start
            pct = 100 * checked / (total * (total - 1) / 2)
            print(f"  Progress: {pct:.1f}% | Best so far: {best_count}/{num_discs} | "
                  f"Time: {elapsed:.1f}s", end='\r')

    print()
    return best_combo, best_count


# ─────────────────────────────────────────────
# SAVE & VISUALIZE
# ─────────────────────────────────────────────

def save_policy(best_cells, grid_positions, disc_positions, coverage_count):
    best_positions = [grid_positions[c] for c in best_cells]
    uav_names = ["uav1", "uav2", "uav3"]

    policy_data = {
        "best_positions": best_positions,
        "coverage": coverage_count,
        "disc_positions": disc_positions,
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
    print(f"Optimal positions:")
    for i, (x, y) in enumerate(best_positions):
        print(f"  UAV{i+1}: ({x:.2f}, {y:.2f})")
    print(f"Coverage: {coverage_count}/{len(disc_positions)} discs "
          f"({100*coverage_count/len(disc_positions):.1f}%)")


def plot_coverage(best_cells, grid_positions, disc_positions, coverage_count):
    best_positions = [grid_positions[c] for c in best_cells]
    fig, ax = plt.subplots(figsize=(10, 10))

    # All discs
    for dx, dy in disc_positions:
        ax.scatter(dx, dy, color='gray', s=60, zorder=2)

    colors = ['red', 'blue', 'green']
    covered = set()
    for i, (ux, uy) in enumerate(best_positions):
        circle = plt.Circle((ux, uy), COVERAGE_RADIUS,
                             alpha=0.25, color=colors[i],
                             label=f'UAV{i+1} ({ux:.1f}, {uy:.1f})')
        ax.add_patch(circle)
        ax.scatter(ux, uy, color=colors[i], marker='X', s=300, zorder=5)
        for j, (dx, dy) in enumerate(disc_positions):
            if np.sqrt((dx-ux)**2 + (dy-uy)**2) <= COVERAGE_RADIUS:
                covered.add(j)

    for j in covered:
        dx, dy = disc_positions[j]
        ax.scatter(dx, dy, color='lime', s=100, zorder=4)

    ax.set_xlim(GRID_MIN - 5, GRID_MAX + 5)
    ax.set_ylim(GRID_MIN - 5, GRID_MAX + 5)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_title(f'Optimal Drone Placement — {coverage_count}/{len(disc_positions)} discs covered '
                 f'({100*coverage_count/len(disc_positions):.0f}%)',
                 fontsize=14)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.legend(loc='upper right', fontsize=11)
    plt.tight_layout()
    plt.savefig(COVERAGE_PLOT, dpi=150)
    print(f"Saved coverage plot to {COVERAGE_PLOT}")


# ─────────────────────────────────────────────
# POLICY LOADER (used by policy_executor.py)
# ─────────────────────────────────────────────

def load_policy(policy_path=POLICY_SAVE_PATH):
    """Load saved policy and return a callable for policy_executor.py."""
    with open(policy_path, "rb") as f:
        data = pickle.load(f)
    best_positions = data["best_positions"]
    uav_names = ["uav1", "uav2", "uav3"]

    def trained_policy(state, disc_positions):
        return {uav: pos for uav, pos in zip(uav_names, best_positions)}

    print(f"Loaded policy — coverage: {data['coverage']}/{len(data['disc_positions'])} discs")
    return trained_policy


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Load disc positions
    if not os.path.exists(DISC_POSITIONS_PATH):
        print(f"ERROR: {DISC_POSITIONS_PATH} not found.")
        exit(1)

    with open(DISC_POSITIONS_PATH) as f:
        data = json.load(f)
    disc_positions = [(p[0], p[1]) for p in data["positions"]]
    print(f"Loaded {len(disc_positions)} disc positions")

    # Build grid
    grid_positions = build_grid()
    print(f"Grid: {GRID_SIZE}x{GRID_SIZE} = {len(grid_positions)} candidate positions")
    print(f"Coverage radius: {COVERAGE_RADIUS}m")

    # Precompute coverage
    print("\nPrecomputing coverage matrix...")
    coverage_matrix = compute_coverage_matrix(grid_positions, disc_positions)

    # Search
    print("\nSearching for optimal placement...")
    start = time.time()
    best_cells, best_count = exhaustive_search(coverage_matrix, NUM_DRONES, len(disc_positions))
    elapsed = time.time() - start
    print(f"\nSearch complete in {elapsed:.1f}s")

    # Save and plot
    save_policy(best_cells, grid_positions, disc_positions, best_count)
    plot_coverage(best_cells, grid_positions, disc_positions, best_count)

    print("\nDone! Copy policy.pkl to your ROS package scripts/ folder.")
