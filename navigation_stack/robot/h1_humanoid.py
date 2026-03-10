"""
H1 Humanoid Robot with RL Walking Policy.
Uses the official Isaac Sim H1FlatTerrainPolicy to make the robot walk naturally.
"""

import numpy as np
import omni.timeline
from omni.isaac.core.utils.rotations import quat_to_euler_angles

# --- USE OFFICIAL POLICY ---
try:
    from isaacsim.robot.policy.examples.robots.h1 import H1FlatTerrainPolicy
except ImportError:
    # Fallback for older versions or different python paths
    print("❌ Critical: Could not import H1FlatTerrainPolicy. Ensure isaacsim.robot.policy extension is enabled.")
    H1FlatTerrainPolicy = None

from isaacsim.storage.native import get_assets_root_path

class H1Humanoid:
    """
    H1 humanoid that uses a Pre-trained RL policy to walk to a target
    with an optional start delay.
    """
    
    def __init__(
        self,
        world,
        spawn_position,
        walk_distance=5.0,
        walk_speed=0.75,
        walk_direction=None,
        start_delay=0.0  # <--- NEW PARAMETER
    ):
        self.world = world
        self.spawn_position = np.array(spawn_position)
        self.walk_speed = walk_speed
        self.start_delay = start_delay
        self.time_elapsed = 0.0 # Track simulation time
        
        # Calculate Target Position
        if walk_direction is None or np.allclose(walk_direction, 0):
            dir_vec = np.array([0, 1, 0])
        else:
            dir_vec = np.array(walk_direction[:3])
            dir_vec = dir_vec / np.linalg.norm(dir_vec)

        self.target_position = self.spawn_position + (dir_vec * walk_distance)
        self.is_stationary = (walk_distance < 0.1)

        self.prim_path = f"/World/H1_{id(self)}"
        self.policy = None
        self.initialized = False
        
        print(f"H1 [{self.prim_path}] Goal: {self.target_position} | Delay: {self.start_delay}s")

    def spawn(self):
        """Loads the H1 Policy and Asset."""
        if H1FlatTerrainPolicy is None:
            return False
            
        try:
            assets_root_path = get_assets_root_path()
            if assets_root_path is None:
                return False

            usd_path = assets_root_path + "/Isaac/Robots/Unitree/H1/h1.usd"

            self.policy = H1FlatTerrainPolicy(
                prim_path=self.prim_path,
                name=f"H1_{id(self)}",
                usd_path=usd_path,
                position=self.spawn_position
            )
            
            return True

        except Exception as e:
            print(f"❌ Failed to spawn H1: {e}")
            return False

    def on_timeline_event(self, event):
        """Reset internal state on stop/play."""
        if event.type == int(omni.timeline.TimelineEventType.STOP):
            self.initialized = False
            self.time_elapsed = 0.0 # Reset timer on stop
        if event.type == int(omni.timeline.TimelineEventType.PLAY):
            self.time_elapsed = 0.0 # Ensure 0 on play

    def on_physics_step(self, step_size):
        """
        Called every physics step. 
        """
        if self.policy is None:
            return

        # --- 1. Accumulate Time ---
        self.time_elapsed += step_size

        # --- 2. Late Initialization ---
        if not self.initialized:
            try:
                self.policy.initialize()
                self.policy.post_reset()
                self.initialized = True
            except Exception:
                return 

        # --- 3. Check Delay Logic ---
        # If we haven't reached the start time, command 0 velocity (stand still)
        if self.time_elapsed < self.start_delay:
            # Stand still and balance
            self.policy.forward(step_size, [0.0, 0.0, 0.0])
            return

        # --- 4. Logic: Move to Goal (Standard) ---
        if self.is_stationary:
            command = [0.0, 0.0, 0.0]
        else:
            current_pos, current_rot = self.policy.robot.get_world_pose()
            vec_to_target = self.target_position[:2] - current_pos[:2]
            dist = np.linalg.norm(vec_to_target)

            if dist < 0.3:
                command = [0.0, 0.0, 0.0]
                self.is_stationary = True
            else:
                desired_yaw = np.arctan2(vec_to_target[1], vec_to_target[0])
                r, p, y = quat_to_euler_angles(current_rot)
                current_yaw = y

                yaw_error = desired_yaw - current_yaw
                yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi

                kp_turn = 1.5
                omega_z = np.clip(yaw_error * kp_turn, -1.0, 1.0)
                
                forward_speed = self.walk_speed
                if abs(yaw_error) > 0.5: 
                    forward_speed = 0.1 

                command = [forward_speed, 0.0, omega_z]

        # Apply Command
        self.policy.forward(step_size, command)
