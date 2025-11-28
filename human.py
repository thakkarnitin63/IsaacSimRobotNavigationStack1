# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np

# --- 1. IMPORT SimulationApp FIRST ---
from isaacsim import SimulationApp

# --- 2. START THE APP ---
# We don't need to manually enable extensions, as the H1 policy
# is part of the core isaacsim python library.
print("Starting SimulationApp...")
simulation_app = SimulationApp({"headless": False})

# --- 3. ALL OTHER IMPORTS GO HERE ---
import carb
import omni.timeline
from omni.isaac.core import World
from omni.isaac.core.utils.stage import get_current_stage
from isaacsim.storage.native import get_assets_root_path

# This is the H1 robot policy class from the example you found
try:
    from isaacsim.robot.policy.examples.robots.h1 import H1FlatTerrainPolicy
except ImportError as e:
    carb.log_error(f"Failed to import H1FlatTerrainPolicy: {e}")
    carb.log_error("This policy may not be available in your Isaac Sim version, or an extension is missing.")
    simulation_app.close()
    exit()

# --- 4. DEFINE THE CONTROLLER CLASS ---
class H1Controller:
    """
    A simple controller to make the H1 robot walk from A to B.
    It holds the robot's state and implements the on_physics_step logic.
    """
    def __init__(self, h1_robot, physics_dt, start_pos, end_pos):
        self.h1 = h1_robot
        self.physics_dt = physics_dt
        self._physics_ready = False
        
        self._step_count = 0
        self._forward_speed = 0.75  # m/s (from your example)
        
        # Calculate distance and time needed
        self._start_pos = np.array(start_pos)
        self._end_pos = np.array(end_pos)
        self._target_distance = np.linalg.norm(self._end_pos - self._start_pos)
        
        # Calculate how many physics steps we need to walk
        time_needed = self._target_distance / self._forward_speed
        self._target_steps = int(time_needed / self.physics_dt)

        print(f"Robot will walk {self._target_distance:.2f} meters.")
        print(f"This will take {time_needed:.2f} seconds, or {self._target_steps} physics steps.")

    def on_physics_step(self, step_size):
        """
        This function is called every physics frame.
        """
        if self._physics_ready:
            # By default, the command is to stop
            command = [0.0, 0.0, 0.0]
            
            # If we haven't reached our target step count, command "move forward"
            if self._step_count < self._target_steps:
                command = [self._forward_speed, 0.0, 0.0]
                self._step_count += 1
            
            # Send the command (either "move" or "stop") to the policy
            self.h1.forward(step_size, command)
            
            if self._step_count == self._target_steps:
                print("Location B reached. Stopping.")
                self._step_count += 1 # Increment once more to avoid re-printing
                
        else:
            # This block runs once when physics starts
            print("Physics is ready. Initializing H1 robot policy...")
            self._physics_ready = True
            self.h1.initialize()
            self.h1.post_reset()
            self.h1.robot.set_joints_default_state(self.h1.default_pos)

    def on_timeline_event(self, event):
        """
        Resets the _physics_ready flag when the timeline is played/stopped.
        This is good practice from the example file.
        """
        if self.h1:
            self._physics_ready = False


# --- 5. SETUP THE SCENE ---

# Define our A and B locations
LOCATION_A = [0.0, 0.0, 1.05]
LOCATION_B = [5.0, 0.0, 1.05] # 5 meters forward
PHYSICS_DT = 1.0 / 200.0      # Physics rate from your example

# Create the World with the correct physics DT
world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT)
stage = get_current_stage()

# Add a ground plane
world.scene.add_default_ground_plane(
    z_position=0,
    static_friction=0.2,
    dynamic_friction=0.2,
    restitution=0.01,
)

# --- 6. SPAWN THE H1 ROBOT ---
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder. Exiting.")
    simulation_app.close()
    exit()

print("Spawning H1 Robot...")
h1_robot_policy = H1FlatTerrainPolicy(
    prim_path="/World/H1",
    name="H1",
    usd_path=assets_root_path + "/Isaac/Robots/Unitree/H1/h1.usd",
    position=np.array(LOCATION_A), # Spawn at Location A
)

# --- 7. SETUP THE CONTROLLER AND CALLBACKS ---
controller = H1Controller(h1_robot_policy, PHYSICS_DT, LOCATION_A, LOCATION_B)

# Add the physics step callback
world.add_physics_callback("physics_step", controller.on_physics_step)

# Add the timeline event callback
timeline = omni.timeline.get_timeline_interface()
timeline_event_sub = timeline.get_timeline_event_stream().create_subscription_to_pop_by_type(
    int(omni.timeline.TimelineEventType.PLAY), controller.on_timeline_event
)

# --- 8. RUN THE SIMULATION ---
world.play()
print("\nSimulation is running. Robot will walk from A to B.")

while simulation_app.is_running():
    world.step(render=True)
    
    if not world.is_playing():
        break # Exit the loop if simulation stops

# Cleanup
timeline_event_sub = None # Unsubscribe
simulation_app.close()
print("Simulation closed.")

