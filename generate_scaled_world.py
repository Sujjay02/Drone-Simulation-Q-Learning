#!/usr/bin/env python3
"""
generate_scaled_world.py
------------------------
Generates a Gazebo world file with disc positions scaled from the Python
deep_q.py simulation to the real 100x100m Gazebo world.

Python sim specs:
  - Area: 0 to 4m (4x4m)
  - Coverage radius: 1.2m
  - 25 disc positions

Gazebo world specs:
  - Area: -50 to 50m (100x100m)
  - Scale factor: 25x
  - Coverage radius: 30m (at 21m altitude with 110 degree FOV)
  - Drone height: 21m

Run: python3 generate_scaled_world.py
"""

import json
import numpy as np

# ─────────────────────────────────────────────
# ORIGINAL PYTHON SIM DISC POSITIONS (from deep_q.py)
# ─────────────────────────────────────────────
PYTHON_POSITIONS = [
    (1.73, 2.91), (1.33, 2.80), (1.25, 0.83), (2.59, 2.91),
    (1.64, 1.15), (2.01, 2.17), (1.60, 1.70), (0.47, 2.05),
    (1.81, 1.96), (1.64, 1.28), (2.01, 1.86), (2.94, 1.07),
    (1.26, 2.92), (1.56, 1.29), (2.03, 2.67), (1.89, 2.02),
    (1.76, 0.98), (1.74, 2.00), (1.69, 2.01), (2.45, 1.56),
    (0.39, 2.64), (3.27, 2.07), (3.27, 3.89), (2.89, 2.60),
    (0.60, 3.20),
]

# ─────────────────────────────────────────────
# SCALING PARAMETERS
# ─────────────────────────────────────────────
PYTHON_AREA_MIN  = 0.0
PYTHON_AREA_MAX  = 4.0
GAZEBO_AREA_MIN  = -50.0
GAZEBO_AREA_MAX  =  50.0
SCALE_FACTOR     = (GAZEBO_AREA_MAX - GAZEBO_AREA_MIN) / (PYTHON_AREA_MAX - PYTHON_AREA_MIN)  # 25x

# Camera/coverage parameters
CAMERA_FOV_RAD   = 1.92        # from f450.sdf.jinja
FLIGHT_ALTITUDE  = 21.0        # meters
COVERAGE_RADIUS  = FLIGHT_ALTITUDE * np.tan(CAMERA_FOV_RAD / 2)  # ~30m

print(f"Scale factor: {SCALE_FACTOR}x")
print(f"Camera FOV: {np.degrees(CAMERA_FOV_RAD):.1f} degrees")
print(f"Flight altitude: {FLIGHT_ALTITUDE}m")
print(f"Ground coverage radius: {COVERAGE_RADIUS:.2f}m")

# ─────────────────────────────────────────────
# SCALE POSITIONS
# ─────────────────────────────────────────────
def scale_position(px, py):
    """Scale from Python sim coords to Gazebo world coords."""
    # Center the Python positions (0-4 → -2 to 2), then scale
    gx = (px - (PYTHON_AREA_MAX / 2)) * SCALE_FACTOR
    gy = (py - (PYTHON_AREA_MAX / 2)) * SCALE_FACTOR
    return round(gx, 2), round(gy, 2)

scaled_positions = [scale_position(px, py) for px, py in PYTHON_POSITIONS]

print(f"\nScaled disc positions ({len(scaled_positions)} discs):")
for i, (x, y) in enumerate(scaled_positions):
    print(f"  disc_{i+1:02d}: ({x:7.2f}, {y:7.2f})")

# ─────────────────────────────────────────────
# SAVE DISC POSITIONS JSON
# ─────────────────────────────────────────────
disc_data = {
    "positions": [[x, y, 0.01] for x, y in scaled_positions],
    "area_size": GAZEBO_AREA_MAX - GAZEBO_AREA_MIN,
    "coverage_radius": round(COVERAGE_RADIUS, 2),
    "flight_altitude": FLIGHT_ALTITUDE,
    "scale_factor": SCALE_FACTOR,
    "camera_fov_deg": round(np.degrees(CAMERA_FOV_RAD), 1),
    "num_discs": len(scaled_positions)
}

with open("disc_positions.json", "w") as f:
    json.dump(disc_data, f, indent=2)
print(f"\nSaved disc_positions.json")

# ─────────────────────────────────────────────
# GENERATE WORLD FILE
# ─────────────────────────────────────────────
def disc_model(index, x, y, z=0.01):
    return f"""
    <model name="disc_{index}">
      <static>true</static>
      <pose>{x} {y} {z} 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <pose>0 0 0 0 0 0</pose>
          <geometry>
            <cylinder>
              <radius>0.5</radius>
              <length>0.05</length>
            </cylinder>
          </geometry>
          <max_contacts>10</max_contacts>
          <surface>
            <contact><ode/></contact>
            <bounce/>
            <friction><torsional><ode/></torsional><ode/></friction>
          </surface>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.5</radius>
              <length>0.05</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>1 0 0 1</ambient>
            <diffuse>1 0 0 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
            <emissive>0.4 0 0 1</emissive>
          </material>
        </visual>
        <self_collide>0</self_collide>
        <enable_wind>0</enable_wind>
        <kinematic>0</kinematic>
      </link>
    </model>"""

discs_sdf = "\n".join(disc_model(i+1, x, y) for i, (x, y) in enumerate(scaled_positions))

world_sdf = f"""<?xml version='1.0' encoding='UTF-8'?>
<sdf version="1.7">
  <world name="disc_world">

    <plugin name="mrs_gazebo_static_transform_republisher_plugin"
            filename="libMrsGazeboCommonResources_StaticTransformRepublisher.so"/>

    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model>
      <latitude_deg>47.3977</latitude_deg>
      <longitude_deg>8.54559</longitude_deg>
      <elevation>0</elevation>
      <heading_deg>0</heading_deg>
    </spherical_coordinates>

    <physics name="default_physics" default="0" type="ode">
      <ode>
        <solver>
          <type>quick</type>
          <iters>10</iters>
          <sor>1.3</sor>
          <use_dynamic_moi_rescaling>0</use_dynamic_moi_rescaling>
        </solver>
        <constraints>
          <cfm>0</cfm>
          <erp>0.2</erp>
          <contact_max_correcting_vel>1000</contact_max_correcting_vel>
          <contact_surface_layer>0.001</contact_surface_layer>
        </constraints>
      </ode>
      <max_step_size>0.004</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>250</real_time_update_rate>
    </physics>

    <scene>
      <shadows>false</shadows>
      <ambient>0.4 0.4 0.4 1</ambient>
      <background>0.7 0.7 0.7 1</background>
    </scene>

    <light name="sun" type="directional">
      <pose>0 0 1000 0.4 0.2 0</pose>
      <diffuse>1 1 1 1</diffuse>
      <specular>0.6 0.6 0.6 1</specular>
      <direction>0.1 0.1 -0.9</direction>
      <attenuation>
        <range>20</range>
        <constant>0.5</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <cast_shadows>1</cast_shadows>
    </light>

    <model name="ground_plane">
      <static>1</static>
      <link name="link">
        <collision name="collision">
          <pose>0 0 0 0 0 0</pose>
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>300 300</size>
            </plane>
          </geometry>
          <surface>
            <friction><ode><mu>1</mu><mu2>1</mu2></ode></friction>
          </surface>
          <max_contacts>10</max_contacts>
        </collision>
        <visual name="visual">
          <pose>0 0 0 0 0 0</pose>
          <cast_shadows>0</cast_shadows>
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>300 300</size>
            </plane>
          </geometry>
          <material>
            <script>
              <uri>file://media/materials/scripts/gazebo.material</uri>
              <n>Gazebo/Grey</n>
            </script>
          </material>
        </visual>
      </link>
    </model>

    <plugin name="mrs_gazebo_rviz_cam_synchronizer"
            filename="libMrsGazeboCommonResources_RvizCameraSynchronizer.so">
      <target_frame_id>gazebo_user_camera</target_frame_id>
      <world_origin_frame_id>uav1/gps_origin</world_origin_frame_id>
      <frame_to_follow>uav1</frame_to_follow>
    </plugin>

    <gravity>0 0 -9.8066</gravity>
    <magnetic_field>6e-06 2.3e-05 -4.2e-05</magnetic_field>
    <atmosphere type="adiabatic"/>
    <wind/>

    <!-- Bird's eye camera view at 150m -->
    <gui fullscreen="0">
      <camera name="camera">
        <pose>0 0 150 0 1.5707 0</pose>
        <view_controller>orbit</view_controller>
        <projection_type>perspective</projection_type>
      </camera>
    </gui>

    <!--
      25 Discs scaled from Python deep_q.py simulation
      Scale factor: {SCALE_FACTOR}x
      Original area: {PYTHON_AREA_MIN}-{PYTHON_AREA_MAX}m
      Gazebo area: {GAZEBO_AREA_MIN}-{GAZEBO_AREA_MAX}m
      Flight altitude: {FLIGHT_ALTITUDE}m
      Ground coverage radius: {COVERAGE_RADIUS:.1f}m
    -->
{discs_sdf}

  </world>
</sdf>
"""

with open("disc_world.world", "w") as f:
    f.write(world_sdf)

print(f"Generated disc_world.world")
print(f"\nUpdate policy_executor.py with:")
print(f"  FLIGHT_ALTITUDE  = {FLIGHT_ALTITUDE}")
print(f"  COVERAGE_RADIUS  = {COVERAGE_RADIUS:.1f}")
print(f"  AREA_X_MIN = AREA_Y_MIN = {GAZEBO_AREA_MIN}")
print(f"  AREA_X_MAX = AREA_Y_MAX = {GAZEBO_AREA_MAX}")
print(f"\nUpdate train_policy.py with:")
print(f"  GRID_MIN        = {GAZEBO_AREA_MIN}")
print(f"  GRID_MAX        = {GAZEBO_AREA_MAX}")
print(f"  COVERAGE_RADIUS = {COVERAGE_RADIUS:.1f}")
