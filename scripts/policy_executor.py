#!/usr/bin/env python3
"""
policy_executor.py
------------------
Phase 1: Policy Execution Framework for 3-drone disc coverage.

This ROS node:
  - Reads real-time positions of uav1, uav2, uav3 via odometry
  - Feeds positions into a policy function
  - Sends movement commands to each drone via MRS goto service

To swap in a trained policy: replace the `policy()` function only.
Everything else (position reading, command sending, loop) stays the same.

Usage:
  rosrun policy_executor policy_executor.py

Or add to session.yml as a new tmux pane after takeoff.
"""

import rospy
import random
import math
import json
import os
import numpy as np

from nav_msgs.msg import Odometry
from mrs_msgs.srv import Vec4, Vec4Request
from std_msgs.msg import String


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# Area bounds (must match your disc_world.world)
AREA_X_MIN = -45.0
AREA_X_MAX =  45.0
AREA_Y_MIN = -45.0
AREA_Y_MAX =  45.0

# Flight altitude for all drones (meters)
FLIGHT_ALTITUDE = 5.0

# How often the policy is queried and commands are sent (seconds)
POLICY_RATE_HZ = 0.2  # every 5 seconds — give drones time to reach targets

# Coverage radius per drone (meters) — used for reward calculation
COVERAGE_RADIUS = 5.0

# Path to disc positions JSON (from your DiscWorld repo)
DISC_POSITIONS_PATH = os.path.expanduser(
    "~/user_ros_workspace/DiscWorld/disc_positions.json"
)

# UAV names
UAV_NAMES = ["uav1", "uav2", "uav3"]


# ─────────────────────────────────────────────
# LOAD DISC POSITIONS
# ─────────────────────────────────────────────

def load_disc_positions():
    """Load the 25 disc positions from the JSON file."""
    try:
        with open(DISC_POSITIONS_PATH, "r") as f:
            data = json.load(f)
        positions = [(p[0], p[1]) for p in data["positions"]]
        rospy.loginfo(f"[PolicyExecutor] Loaded {len(positions)} disc positions.")
        return positions
    except Exception as e:
        rospy.logwarn(f"[PolicyExecutor] Could not load disc positions: {e}")
        return []


# ─────────────────────────────────────────────
# POLICY FUNCTION
# ─────────────────────────────────────────────
# THIS IS THE ONLY FUNCTION YOU NEED TO REPLACE
# when you have a trained policy.
#
# Input:
#   state — dict with keys "uav1", "uav2", "uav3"
#           each value is (x, y, z) current position
#
# Output:
#   actions — dict with keys "uav1", "uav2", "uav3"
#             each value is (x, y) target position
#             (z is handled automatically as FLIGHT_ALTITUDE)
# ─────────────────────────────────────────────

def policy(state, disc_positions):
    """
    RANDOM POLICY (placeholder).
    Sends each drone to a random position in the area.

    Replace this function body with your trained policy.
    The function signature must stay the same.

    Example trained policy swap-in:
        import pickle
        with open("trained_policy.pkl", "rb") as f:
            model = pickle.load(f)
        obs = state_to_observation(state)
        actions_array = model.predict(obs)
        return observation_to_actions(actions_array)
    """
    actions = {}
    for uav in UAV_NAMES:
        x = random.uniform(AREA_X_MIN, AREA_X_MAX)
        y = random.uniform(AREA_Y_MIN, AREA_Y_MAX)
        actions[uav] = (x, y)
    return actions


# ─────────────────────────────────────────────
# REWARD CALCULATOR
# ─────────────────────────────────────────────

def calculate_coverage(drone_positions, disc_positions):
    """
    Calculate how many discs are covered by any drone.
    A disc is covered if it's within COVERAGE_RADIUS of any drone.

    Returns:
        covered_count  — number of discs covered
        coverage_ratio — fraction of discs covered (0.0 to 1.0)
        covered_discs  — list of indices of covered discs
    """
    if not disc_positions:
        return 0, 0.0, []

    covered = set()
    for disc_idx, (dx, dy) in enumerate(disc_positions):
        for uav, (ux, uy, _) in drone_positions.items():
            dist = math.sqrt((dx - ux)**2 + (dy - uy)**2)
            if dist <= COVERAGE_RADIUS:
                covered.add(disc_idx)
                break  # disc is covered, no need to check other drones

    covered_count = len(covered)
    coverage_ratio = covered_count / len(disc_positions)
    return covered_count, coverage_ratio, list(covered)


# ─────────────────────────────────────────────
# POLICY EXECUTOR NODE
# ─────────────────────────────────────────────

class PolicyExecutor:

    def __init__(self):
        rospy.init_node("policy_executor", anonymous=False)
        rospy.loginfo("[PolicyExecutor] Initializing...")

        # Current drone positions: {uav_name: (x, y, z)}
        self.positions = {
            "uav1": (0.0, 0.0, 0.0),
            "uav2": (0.0, 0.0, 0.0),
            "uav3": (0.0, 0.0, 0.0),
        }
        self.positions_received = {uav: False for uav in UAV_NAMES}

        # Load disc positions for reward calculation
        self.disc_positions = load_disc_positions()

        # Step counter and coverage log
        self.step = 0
        self.coverage_log = []

        # Subscribe to odometry for each drone
        for uav in UAV_NAMES:
            rospy.Subscriber(
    		f"/{uav}/hw_api/odometry",
                Odometry,
                self._odom_callback,
                callback_args=uav,
                queue_size=1
            )

        # Wait for MRS goto services
        rospy.loginfo("[PolicyExecutor] Waiting for goto services...")
        self.goto_services = {}
        for uav in UAV_NAMES:
            service_name = f"/{uav}/control_manager/goto"
            rospy.wait_for_service(service_name, timeout=30.0)
            self.goto_services[uav] = rospy.ServiceProxy(service_name, Vec4)
        rospy.loginfo("[PolicyExecutor] All goto services ready.")

        # Publisher for coverage status
        self.coverage_pub = rospy.Publisher(
            "/policy_executor/coverage_status",
            String,
            queue_size=1
        )

        rospy.loginfo("[PolicyExecutor] Ready. Starting policy loop.")

    def _odom_callback(self, msg, uav_name):
        """Update drone position from odometry."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        self.positions[uav_name] = (x, y, z)
        self.positions_received[uav_name] = True

    def _send_goto(self, uav_name, x, y, z=None, heading=0.0):
        """
        Send a goto command to a drone using MRS control_manager/goto.
        Vec4 format: [x, y, z, heading]
        """
        if z is None:
            z = FLIGHT_ALTITUDE

        try:
            req = Vec4Request()
            req.goal = [x, y, z, heading]
            resp = self.goto_services[uav_name](req)
            if resp.success:
                rospy.loginfo(
                    f"[PolicyExecutor] {uav_name} → goto ({x:.1f}, {y:.1f}, {z:.1f})"
                )
            else:
                rospy.logwarn(
                    f"[PolicyExecutor] {uav_name} goto rejected: {resp.message}"
                )
        except rospy.ServiceException as e:
            rospy.logerr(f"[PolicyExecutor] {uav_name} goto service failed: {e}")

    def _all_positions_received(self):
        """Check if we've received at least one odometry message from all drones."""
        return all(self.positions_received.values())

    def run(self):
        """Main policy execution loop."""
        rate = rospy.Rate(POLICY_RATE_HZ)

        # Wait until all drones are reporting positions
        rospy.loginfo("[PolicyExecutor] Waiting for all drone positions...")
        while not rospy.is_shutdown():
            if self._all_positions_received():
                break
            rospy.loginfo_throttle(5.0, "[PolicyExecutor] Still waiting for odometry...")
            rate.sleep()

        rospy.loginfo("[PolicyExecutor] All positions received. Starting policy execution.")

        while not rospy.is_shutdown():
            self.step += 1

            # ── 1. Build state from current positions ──
            state = dict(self.positions)

            # ── 2. Calculate current coverage before action ──
            covered, ratio, covered_ids = calculate_coverage(
                self.positions, self.disc_positions
            )
            rospy.loginfo(
                f"[PolicyExecutor] Step {self.step} | "
                f"Coverage: {covered}/{len(self.disc_positions)} discs "
                f"({ratio*100:.1f}%)"
            )

            # Publish coverage status
            status_msg = (
                f"step={self.step} "
                f"covered={covered} "
                f"total={len(self.disc_positions)} "
                f"ratio={ratio:.3f} "
                f"uav1={self.positions['uav1']} "
                f"uav2={self.positions['uav2']} "
                f"uav3={self.positions['uav3']}"
            )
            self.coverage_pub.publish(status_msg)
            self.coverage_log.append({
                "step": self.step,
                "covered": covered,
                "ratio": ratio,
                "positions": dict(self.positions)
            })

            # ── 3. Query policy for actions ──
            actions = policy(state, self.disc_positions)

            # ── 4. Send goto commands ──
            for uav, (tx, ty) in actions.items():
                # Clamp to area bounds just in case
                tx = max(AREA_X_MIN, min(AREA_X_MAX, tx))
                ty = max(AREA_Y_MIN, min(AREA_Y_MAX, ty))
                self._send_goto(uav, tx, ty)

            rospy.loginfo(
                f"[PolicyExecutor] Actions: "
                f"uav1→({actions['uav1'][0]:.1f},{actions['uav1'][1]:.1f}) "
                f"uav2→({actions['uav2'][0]:.1f},{actions['uav2'][1]:.1f}) "
                f"uav3→({actions['uav3'][0]:.1f},{actions['uav3'][1]:.1f})"
            )

            rate.sleep()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    try:
        executor = PolicyExecutor()
        executor.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("[PolicyExecutor] Shutting down.")
